import { useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Download, FileSpreadsheet, Loader2 } from 'lucide-react'
import { exportNodeToPng, type ExportPngOptions } from '@/lib/export-png'
import { exportToCsv, type CsvColumn } from '@/lib/export-csv'

interface ExportableCardProps {
  title?: string
  filename: string
  children: React.ReactNode
  headerClassName?: string
  contentClassName?: string
  exportPngOptions?: ExportPngOptions
  csvData?: Record<string, unknown>[]
  csvColumns?: CsvColumn[]
  csvFilename?: string
}

export default function ExportableCard({
  title,
  filename,
  children,
  headerClassName,
  contentClassName,
  exportPngOptions,
  csvData,
  csvColumns,
  csvFilename,
}: ExportableCardProps) {
  const contentRef = useRef<HTMLDivElement>(null)
  const [exporting, setExporting] = useState(false)

  async function handleExportPng() {
    if (!contentRef.current || exporting) return
    setExporting(true)
    try {
      await exportNodeToPng(contentRef.current, filename, exportPngOptions)
    } finally {
      setExporting(false)
    }
  }

  function handleExportCsv() {
    if (!csvData || !csvColumns) return
    exportToCsv(csvData, csvColumns, csvFilename ?? filename)
  }

  const hasCsv = csvData && csvColumns && csvData.length > 0

  const exportButtons = (
    <div className="inline-flex items-center gap-2">
      <button
        onClick={handleExportPng}
        disabled={exporting}
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
        title="Export as PNG"
      >
        {exporting ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Download className="h-4 w-4" />
        )}
        PNG
      </button>
      {hasCsv && (
        <button
          onClick={handleExportCsv}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          title="Export as CSV"
        >
          <FileSpreadsheet className="h-4 w-4" />
          CSV
        </button>
      )}
    </div>
  )

  return (
    <Card>
      {title ? (
        <CardHeader className={headerClassName ?? 'pb-2'}>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">{title}</CardTitle>
            {exportButtons}
          </div>
        </CardHeader>
      ) : (
        <div className="flex justify-end px-4 pt-3">
          {exportButtons}
        </div>
      )}
      <CardContent ref={contentRef} className={contentClassName}>
        {children}
      </CardContent>
    </Card>
  )
}
