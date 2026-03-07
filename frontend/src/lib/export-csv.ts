export interface CsvColumn {
  header: string
  accessor: (row: Record<string, unknown>) => string | number
}

function escapeCsvValue(value: string | number): string {
  const str = String(value)
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`
  }
  return str
}

export function buildCsvString(
  data: Record<string, unknown>[],
  columns: CsvColumn[]
): string {
  const header = columns.map((c) => escapeCsvValue(c.header)).join(',')
  const rows = data.map((row) =>
    columns.map((c) => escapeCsvValue(c.accessor(row))).join(',')
  )
  return [header, ...rows].join('\n')
}

export function exportToCsv(
  data: Record<string, unknown>[],
  columns: CsvColumn[],
  filename: string
): void {
  const csv = buildCsvString(data, columns)
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename.endsWith('.csv') ? filename : `${filename}.csv`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}
