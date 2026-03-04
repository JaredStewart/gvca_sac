import { useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Download, Loader2 } from 'lucide-react'
import { exportNodeToPng } from '@/lib/export-png'

interface ExportableCardProps {
  title?: string
  filename: string
  children: React.ReactNode
  headerClassName?: string
  contentClassName?: string
}

export default function ExportableCard({
  title,
  filename,
  children,
  headerClassName,
  contentClassName,
}: ExportableCardProps) {
  const contentRef = useRef<HTMLDivElement>(null)
  const [exporting, setExporting] = useState(false)

  async function handleExport() {
    if (!contentRef.current || exporting) return
    setExporting(true)
    try {
      await exportNodeToPng(contentRef.current, filename)
    } finally {
      setExporting(false)
    }
  }

  return (
    <Card>
      {title && (
        <CardHeader className={headerClassName ?? 'pb-2'}>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">{title}</CardTitle>
            <button
              onClick={handleExport}
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
          </div>
        </CardHeader>
      )}
      {!title && (
        <div className="flex justify-end px-4 pt-3">
          <button
            onClick={handleExport}
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
        </div>
      )}
      <CardContent ref={contentRef} className={contentClassName}>
        {children}
      </CardContent>
    </Card>
  )
}
