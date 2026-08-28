import { Float16Array } from '@petamoriken/float16';
import { convertResponseImage } from './imageHelpers';
import { BufferWithInfo } from './imageBuffer';

const decoders: {
  [version: string]: (
    v0: number,
    v1: number,
    v2: number,
    v3: number
  ) => [number, number, number];
} = {};

decoders['v1'] =
  decoders['v2'] =
  decoders['svdI2v'] =
    (v0, v1, v2, v3) => {
      const r = 49.521 * v0 + 29.0283 * v1 - 23.9673 * v2 - 39.4981 * v3 + 99.9368
      const g = 41.1373 * v0 + 42.4951 * v1 + 24.7349 * v2 - 50.8279 * v3 + 99.8421
      const b = 40.2919 * v0 + 18.9304 * v1 + 30.0236 * v2 - 81.9976 * v3 + 99.5384

      return [r, g, b]
    }

// SDXL latent -> RGB projection. Coefficients are ComfyUI's `latent_formats.SDXL`
// factors/bias (comfy/latent_formats.py), scaled by 255 to match this module's convention.
decoders['sdxl_base_v0.9'] =
  (v0, v1, v2, v3) => {
    const r = 93.0955 * v0 - 64.5915 * v1 + 27.438 * v2 - 80.7075 * v3 + 27.642
    const g = 107.916 * v0 - 1.071 * v1 + 28.3305 * v2 - 63.546 * v3 - 4.4625
    const b = 110.6955 * v0 + 27.234 * v1 - 9.231 * v2 - 55.794 * v3 - 0.2805

    return [r, g, b]
  }

/**
 * Decodes a preview image returned by the gRPC API.
 * The API returns a width * height * 4 array of float16 values.
 * This function decodes it into a width * height * 3 array of bytes (uint8).
 * Currenlty on SD1 is supported
 *
 * @param preview - The raw preview data as a Uint8Array (representing float16s).
 * @param version - The decoder version to use (e.g., 'v1', 'v2', 'svdI2v').
 * @returns A promise that resolves to a `BufferWithInfo` object containing the decoded image.
 */
export async function decodePreview(
  preview: Uint8Array,
  version?: string
): Promise<BufferWithInfo> {
  const intBuffer = new Uint32Array(preview.buffer, 0, 17);
  const [height, width, channels] = intBuffer.slice(6, 9);

  if (!version || !(version in decoders) || channels !== 4)
    return await convertResponseImage(preview);

  const offset = 68;

  // Preview frames streamed mid-sampling can be truncated relative to the header's
  // declared width/height, so clamp to what's actually available instead of trusting it.
  const availableBytes = preview.byteLength - offset
  const declaredFloats = width * height * channels
  const availableFloats = Math.max(0, Math.floor(availableBytes / 2))
  const floatCount = Math.min(declaredFloats, availableFloats - (availableFloats % 4))

  const f16a = new Float16Array(preview.buffer, preview.byteOffset + offset, floatCount)
  const u8c = new Uint8ClampedArray(width * height * 3)

  for (let i = 0; i < f16a.length / 4; i++) {
    [u8c[i * 3], u8c[i * 3 + 1], u8c[i * 3 + 2]] = decoders[version](
      f16a[i * 4],
      f16a[i * 4 + 1],
      f16a[i * 4 + 2],
      f16a[i * 4 + 3]
    );
  }

  return {
    data: u8c!,
    width,
    height,
    channels: 3,
  };
}
