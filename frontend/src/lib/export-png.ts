import { toPng } from 'html-to-image'

const EXPORT_WIDTH = 1200
const EXPORT_HEIGHT = 900

export async function exportNodeToPng(
  node: HTMLElement,
  filename: string
): Promise<void> {
  const dataUrl = await toPng(node, {
    width: EXPORT_WIDTH,
    height: EXPORT_HEIGHT,
    style: {
      width: `${EXPORT_WIDTH}px`,
      height: `${EXPORT_HEIGHT}px`,
      padding: '24px',
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
