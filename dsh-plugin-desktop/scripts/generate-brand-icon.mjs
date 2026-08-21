import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import sharp from 'sharp'

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const source = resolve(packageDirectory, 'build/app-icon.svg')
const destination = resolve(packageDirectory, 'build/app-icon.png')

await sharp(source)
  .resize(1024, 1024)
  .ensureAlpha()
  .toColourspace('rgb16')
  .withIccProfile('srgb')
  .png()
  .toFile(destination)
