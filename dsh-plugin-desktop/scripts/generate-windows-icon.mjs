import { fileURLToPath } from 'node:url'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import sharp from 'sharp'

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const source = resolve(packageDirectory, 'build/app-icon.png')
const destination = resolve(packageDirectory, 'build/app-icon.ico')

// Modern Windows supports PNG payloads inside ICO containers.  Keeping this
// generator beside the source asset makes the desktop shortcut and packaged
// application use the exact same RunDeep mark.
const image = await sharp(source)
  .resize(256, 256, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
  .ensureAlpha()
  .png()
  .toBuffer()

const header = Buffer.alloc(22)
header.writeUInt16LE(0, 0) // reserved
header.writeUInt16LE(1, 2) // ICO type
header.writeUInt16LE(1, 4) // one image
header.writeUInt8(0, 6) // 0 means 256px wide
header.writeUInt8(0, 7) // 0 means 256px high
header.writeUInt8(0, 8)
header.writeUInt8(0, 9)
header.writeUInt16LE(1, 10)
header.writeUInt16LE(32, 12)
header.writeUInt32LE(image.length, 14)
header.writeUInt32LE(22, 18)

await mkdir(dirname(destination), { recursive: true })
await writeFile(destination, Buffer.concat([header, image]))
