import { Float16Array } from '@petamoriken/float16';
import { convertResponseImage } from './imageHelpers';
import { decompress } from './fpzip/decompress';
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
// factors (comfy/latent_formats.py), scaled by 255 to match this module's convention.
// ComfyUI's own bias centers near black; brightened here to match this module's
// v1/v2 mid-gray baseline so the preview is actually visible.
decoders['sdxl_base_v0.9'] =
  (v0, v1, v2, v3) => {
    const r = 93.0955 * v0 - 64.5915 * v1 + 27.438 * v2 - 80.7075 * v3 + 87.642
    const g = 107.916 * v0 - 1.071 * v1 + 28.3305 * v2 - 63.546 * v3 + 55.5375
    const b = 110.6955 * v0 + 27.234 * v1 - 9.231 * v2 - 55.794 * v3 + 59.7195

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

  // The server can FPZIP-compress the preview payload -- same magic number
  // convertResponseImage already checks for final images. Reading compressed bytes as raw
  // float16 without decompressing first produces effectively random values (full-frame static),
  // since the compressed stream doesn't happen to fail loudly, it just decodes as garbage.
  const isCompressed = intBuffer[0] === 1012247;

  let floats: { length: number; [index: number]: number };
  if (isCompressed) {
    floats = await decompress(preview.slice(offset));
  } else {
    // Preview frames streamed mid-sampling can be truncated relative to the header's
    // declared width/height, so clamp to what's actually available instead of trusting it.
    const availableBytes = preview.byteLength - offset
    const declaredFloats = width * height * channels
    const availableFloats = Math.max(0, Math.floor(availableBytes / 2))
    const floatCount = Math.min(declaredFloats, availableFloats - (availableFloats % 4))
    floats = new Float16Array(preview.buffer, preview.byteOffset + offset, floatCount)
  }

  const u8c = new Uint8ClampedArray(width * height * 3)

  for (let i = 0; i < floats.length / 4; i++) {
    [u8c[i * 3], u8c[i * 3 + 1], u8c[i * 3 + 2]] = decoders[version](
      floats[i * 4],
      floats[i * 4 + 1],
      floats[i * 4 + 2],
      floats[i * 4 + 3]
    );
  }

  return {
    data: u8c!,
    width,
    height,
    channels: 3,
  };
}
