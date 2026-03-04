import { useState, useMemo } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { dataApi, type UnifiedResponse } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  ColumnGroupToggle,
  DATA_TABLE_COLUMN_GROUPS,
} from '@/components/data/ColumnGroupToggle'
import { useToast } from '@/components/ui/use-toast'
import { Download, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

// Column definitions
interface ColumnDef {
  id: string
  header: string
  accessor: (row: UnifiedResponse) => string | number | boolean | null | undefined
  group: string
  width?: string
}

const COLUMNS: ColumnDef[] = [
  // Identity columns
  { id: 'respondent_id', header: 'Respondent ID', accessor: (r) => r.respondent_id, group: 'identity', width: 'w-28' },
  { id: 'level', header: 'Level', accessor: (r) => r.level, group: 'identity', width: 'w-32' },

  // Demographics columns
  { id: 'is_minority', header: 'Minority', accessor: (r) => r.is_minority, group: 'demographics', width: 'w-20' },
  { id: 'has_support', header: 'Support', accessor: (r) => r.has_support, group: 'demographics', width: 'w-20' },
  { id: 'years_at_gvca', header: 'Years', accessor: (r) => r.years_at_gvca, group: 'demographics', width: 'w-16' },
  { id: 'is_year_1', header: 'Year 1', accessor: (r) => r.is_year_1, group: 'demographics', width: 'w-16' },
  { id: 'n_parents', header: 'Parents', accessor: (r) => r.n_parents, group: 'demographics', width: 'w-16' },

  // Grammar satisfaction columns (Q1-Q7)
  ...(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'] as const).map((q) => ({
    id: `${q}_Grammar`,
    header: `${q} (G)`,
    accessor: (r: UnifiedResponse) => r.satisfaction?.[q]?.Grammar,
    group: 'satisfaction_grammar',
    width: 'w-20',
  })),

  // Middle satisfaction columns (Q1-Q7)
  ...(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'] as const).map((q) => ({
    id: `${q}_Middle`,
    header: `${q} (M)`,
    accessor: (r: UnifiedResponse) => r.satisfaction?.[q]?.Middle,
    group: 'satisfaction_middle',
    width: 'w-20',
  })),

  // High satisfaction columns (Q1-Q7)
  ...(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'] as const).map((q) => ({
    id: `${q}_High`,
    header: `${q} (H)`,
    accessor: (r: UnifiedResponse) => r.satisfaction?.[q]?.High,
    group: 'satisfaction_high',
    width: 'w-20',
  })),

  // Free response columns
  { id: 'praise_Grammar', header: 'Praise (G)', accessor: (r) => r.free_responses?.praise?.Grammar, group: 'free_responses', width: 'w-48' },
  { id: 'praise_Middle', header: 'Praise (M)', accessor: (r) => r.free_responses?.praise?.Middle, group: 'free_responses', width: 'w-48' },
  { id: 'praise_High', header: 'Praise (H)', accessor: (r) => r.free_responses?.praise?.High, group: 'free_responses', width: 'w-48' },
  { id: 'praise_Generic', header: 'Praise (Gen)', accessor: (r) => r.free_responses?.praise?.Generic, group: 'free_responses', width: 'w-48' },
  { id: 'improvement_Grammar', header: 'Improve (G)', accessor: (r) => r.free_responses?.improvement?.Grammar, group: 'free_responses', width: 'w-48' },
  { id: 'improvement_Middle', header: 'Improve (M)', accessor: (r) => r.free_responses?.improvement?.Middle, group: 'free_responses', width: 'w-48' },
  { id: 'improvement_High', header: 'Improve (H)', accessor: (r) => r.free_responses?.improvement?.High, group: 'free_responses', width: 'w-48' },
  { id: 'improvement_Generic', header: 'Improve (Gen)', accessor: (r) => r.free_responses?.improvement?.Generic, group: 'free_responses', width: 'w-48' },
]

export default function DataTable() {
  const { selectedYear } = useAppStore()
  const { toast } = useToast()

  // Pagination state
  const [page, setPage] = useState(1)
  const perPage = 50

  // Column group visibility state
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(() => {
    const initial = new Set<string>()
    DATA_TABLE_COLUMN_GROUPS.forEach((g) => {
      if (g.defaultVisible) initial.add(g.id)
    })
    return initial
  })

  // Filter state
  const [filters, setFilters] = useState<{
    level?: string
    is_minority?: boolean
    has_support?: boolean
    is_year_1?: boolean
  }>({})

  // Query for unified responses
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['unified-responses', selectedYear, page, perPage, filters],
    queryFn: () =>
      selectedYear
        ? dataApi.getUnifiedResponses(selectedYear, {
            page,
            per_page: perPage,
            ...filters,
          })
        : null,
    enabled: !!selectedYear,
  })

  // Filter visible columns based on column groups
  const visibleColumns = useMemo(() => {
    const visibleColumnIds = new Set<string>()
    DATA_TABLE_COLUMN_GROUPS.forEach((group) => {
      if (visibleGroups.has(group.id)) {
        group.columns.forEach((col) => visibleColumnIds.add(col))
      }
    })
    return COLUMNS.filter((col) => visibleColumnIds.has(col.id))
  }, [visibleGroups])

  // Toggle column group visibility
  const handleToggleGroup = (groupId: string) => {
    setVisibleGroups((prev) => {
      const next = new Set(prev)
      if (next.has(groupId)) {
        next.delete(groupId)
      } else {
        next.add(groupId)
      }
      return next
    })
  }

  // Export mutation for proper loading state
  const exportMutation = useMutation({
    mutationFn: () => {
      if (!selectedYear) throw new Error('No year selected')
      return dataApi.exportData(selectedYear)
    },
    onSuccess: () => {
      toast({
        title: 'Export started',
        description: 'CSV file download should begin shortly',
      })
    },
    onError: (err) => {
      toast({
        title: 'Export failed',
        description: err instanceof Error ? err.message : 'Unknown error',
        variant: 'destructive',
      })
    },
  })

  const handleExport = () => {
    exportMutation.mutate()
  }

  // Format cell value for display
  const formatCellValue = (value: string | number | boolean | null | undefined): string => {
    if (value === null || value === undefined) return '-'
    if (typeof value === 'boolean') return value ? 'Yes' : 'No'
    if (typeof value === 'string' && value.length > 100) return value.substring(0, 100) + '...'
    return String(value)
  }

  if (!selectedYear) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-muted-foreground">Please select a year to view data</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Data Table</h1>
          <p className="text-muted-foreground">
            Unified survey responses for {selectedYear}
            {data?.totalItems != null && (
              <span className="ml-2">({data.totalItems} total)</span>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={exportMutation.isPending}
          >
            {exportMutation.isPending ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Download className="h-4 w-4 mr-2" />
            )}
            {exportMutation.isPending ? 'Exporting...' : 'Export CSV'}
          </Button>
        </div>
      </div>

      {/* Column group toggles */}
      <div className="border rounded-lg p-3 bg-muted/50">
        <ColumnGroupToggle
          groups={DATA_TABLE_COLUMN_GROUPS}
          visibleGroups={visibleGroups}
          onToggle={handleToggleGroup}
        />
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : isError ? (
        <div className="flex items-center justify-center h-64">
          <p className="text-destructive">
            Error loading data: {error instanceof Error ? error.message : 'Unknown error'}
          </p>
        </div>
      ) : !data?.items || data.items.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
          <p className="text-lg">No data found</p>
          <p className="text-sm">Use the "Load Data" button in the header to load survey data</p>
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  {visibleColumns.map((col, idx) => (
                    <TableHead
                      key={col.id}
                      className={cn(
                        col.width,
                        'whitespace-nowrap',
                        idx === 0 && 'sticky left-0 bg-background z-10'
                      )}
                    >
                      {col.header}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.items.map((row) => (
                  <TableRow key={row.id}>
                    {visibleColumns.map((col, idx) => {
                      const value = col.accessor(row)
                      const isFreeResponse = col.group === 'free_responses'

                      return (
                        <TableCell
                          key={col.id}
                          className={cn(
                            col.width,
                            idx === 0 && 'sticky left-0 bg-background z-10 font-mono text-xs',
                            isFreeResponse && 'max-w-[200px]'
                          )}
                        >
                          {isFreeResponse && value ? (
                            <div
                              className="truncate text-xs"
                              title={String(value)}
                            >
                              {formatCellValue(value)}
                            </div>
                          ) : typeof value === 'boolean' ? (
                            <Badge variant={value ? 'default' : 'secondary'} className="text-xs">
                              {value ? 'Yes' : 'No'}
                            </Badge>
                          ) : (
                            <span className="text-sm">{formatCellValue(value)}</span>
                          )}
                        </TableCell>
                      )
                    })}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}

      {/* Pagination */}
      {data && data.totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Page {page} of {data.totalPages} ({data.totalItems} total items)
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.min(data.totalPages, p + 1))}
              disabled={page >= data.totalPages}
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
