/** Product user-data locations, including the one-time Rundeep brand transition. */

import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { posix, win32 } from 'node:path'

export const DESKTOP_PRODUCT_NAME = 'Rundeep'
export const LEGACY_DESKTOP_PRODUCT_NAME = 'RunDeep'

/**
 * Keep an existing installation on its legacy data directory when the new
 * Rundeep directory has not been created yet. New installations always use
 * the Rundeep directory, while an explicit new directory wins when both exist.
 */
export function selectDesktopUserDataDirectory(
  primaryDirectory: string,
  legacyDirectory: string,
  pathExists: (path: string) => boolean = existsSync,
): string {
  if (!pathExists(primaryDirectory) && pathExists(legacyDirectory)) return legacyDirectory
  return primaryDirectory
}

/** Resolve the packaged Desktop user-data location without importing Electron. */
export function defaultDesktopUserDataDirectory(
  platform: NodeJS.Platform = process.platform,
  environment: NodeJS.ProcessEnv = process.env,
  homeDirectory: string = homedir(),
  pathExists: (path: string) => boolean = existsSync,
): string {
  const path = platform === 'win32' ? win32 : posix
  let parentDirectory: string
  if (platform === 'win32') {
    const appData = environment.APPDATA
    if (appData === undefined || appData.length === 0) {
      throw new Error('APPDATA is unavailable; cannot locate Rundeep diagnostics')
    }
    parentDirectory = appData
  } else if (platform === 'darwin') {
    parentDirectory = path.join(homeDirectory, 'Library', 'Application Support')
  } else {
    const config = environment.XDG_CONFIG_HOME
    parentDirectory = config === undefined || config.length === 0
      ? path.join(homeDirectory, '.config')
      : config
  }
  return selectDesktopUserDataDirectory(
    path.join(parentDirectory, DESKTOP_PRODUCT_NAME),
    path.join(parentDirectory, LEGACY_DESKTOP_PRODUCT_NAME),
    pathExists,
  )
}
