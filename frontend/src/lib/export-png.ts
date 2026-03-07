import { toPng } from 'html-to-image'

export interface ExportPngOptions {
  width?: number
  height?: number
  padding?: number
}

export async function exportNodeToPng(
  node: HTMLElement,
  filename: string,
  options?: ExportPngOptions
): Promise<void> {
  const padding = options?.padding ?? 24
  const width = options?.width ?? node.scrollWidth + padding * 2
  const height = options?.height ?? node.scrollHeight + padding * 2

  const dataUrl = await toPng(node, {
    width,
    height,
    style: {
      width: `${width}px`,
      height: `${height}px`,
      padding: `${padding}px`,
      boxSizing: 'border-box',
      background: 'white',
    },
    pixelRatio: 2,
  })

  const link = document.createElement('a')
  link.download = filename.endsWith('.png') ? filename : `${filename}.png`
  link.href = dataUrl
  link.click()
}
