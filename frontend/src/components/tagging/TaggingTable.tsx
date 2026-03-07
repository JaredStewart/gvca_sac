import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { dataApi, tagsApi, taggingApi, type TaggableResponse } from '@/api/client'
import { useAppStore } from '@/stores/app'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { StabilityBadge } from './StabilityBadge'
import { TagCell } from './TagCell'
import {
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Download,
  RefreshCw,
  AlertTriangle,
  Sparkles,
} from 'lucide-react'

const MODEL_OPTIONS = [
  { value: 'gpt-5-nano', label: 'GPT-5 Nano' },
  { value: 'gpt-5-mini', label: 'GPT-5 Mini' },
  { value: 'gpt-5', label: 'GPT-5' },
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
  { value: 'gpt-4o', label: 'GPT-4o' },
]

interface TaggingTableProps {
  onTagToggle?: (responseId: string, tag: string, currentValue: boolean) => void
  loadingTags?: Set<string>
  batchTagAction?: React.ReactNode
}

export function TaggingTable({ onTagToggle, loadingTags, batchTagAction }: TaggingTableProps) {
  const { selectedYear } = useAppStore()
  const queryClient = useQueryClient()
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(100)
  const [levelFilter, setLevelFilter] = useState<string>('all')
  const [questionTypeFilter, setQuestionTypeFilter] = useState<string>('all')
  const [tagFilter, setTagFilter] = useState<string>('all')
  const [selectedModel, setSelectedModel] = useState<string>('gpt-5-nano')
  const [sortBy, setSortBy] = useState<string>('response_id')
  const [keywordMismatchFilter, setKeywordMismatchFilter] = useState<string>('all')
  const [taggingResponseIds, setTaggingResponseIds] = useState<Set<string>>(new Set())
  const [isExporting, setIsExporting] = useState(false)

  // Fetch taxonomy for tag columns
  const { data: taxonomy } = useQuery({
    queryKey: ['taxonomy'],
    queryFn: () => tagsApi.getTaxonomy(),
  })

  // Fetch responses with tags
  const {
    data: responses,
    isLoading,
  } = useQuery({
    queryKey: [
      'free-responses-with-tags',
      selectedYear,
      page,
      perPage,
      levelFilter,
      questionTypeFilter,
      tagFilter,
      sortBy,
      keywordMismatchFilter,
    ],
    queryFn: () =>
      selectedYear
        ? dataApi.getFreeResponsesWithTags(selectedYear, {
            page,
            per_page: perPage,
            level: levelFilter === 'all' ? undefined : levelFilter,
            question_type: questionTypeFilter === 'all' ? undefined : (questionTypeFilter as 'praise' | 'improvement'),
            tag: tagFilter === 'all' ? undefined : tagFilter,
            sort: sortBy,
            has_keyword_mismatch: keywordMismatchFilter === 'all' ? undefined : keywordMismatchFilter === 'mismatch',
          })
        : null,
    enabled: !!selectedYear,
  })

  // Per-response LLM tagging mutation
  const llmTagMutation = useMutation({
    mutationFn: (responseId: string) =>
      taggingApi.tagSingle(selectedYear!, responseId, selectedModel),
    onMutate: (responseId) => {
      setTaggingResponseIds((prev) => new Set(prev).add(responseId))
    },
    onSuccess: (_data, responseId) => {
      setTaggingResponseIds((prev) => {
        const next = new Set(prev)
        next.delete(responseId)
        return next
      })
      queryClient.invalidateQueries({ queryKey: ['free-responses-with-tags'] })
    },
    onError: (error, responseId) => {
      setTaggingResponseIds((prev) => {
        const next = new Set(prev)
        next.delete(responseId)
        return next
      })
      alert(`Failed to tag response: ${error instanceof Error ? error.message : 'Unknown error'}`)
    },
  })

  const handleLLMTag = (responseId: string, hasOverride: boolean) => {
    if (hasOverride) {
      const confirmed = window.confirm(
        'This response has manual tag overrides. LLM tagging will replace them. Continue?'
      )
      if (!confirmed) return
    }
    llmTagMutation.mutate(responseId)
  }

  const handleExport = async () => {
    if (!selectedYear) return
    setIsExporting(true)
    try {
      await dataApi.exportTagging(selectedYear)
    } catch (error) {
      alert(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsExporting(false)
    }
  }

  const toggleSort = () => {
    setSortBy((prev) => (prev === 'stability_asc' ? 'response_id' : 'stability_asc'))
    setPage(1)
  }

  const tags = taxonomy?.tags ?? []

  if (!selectedYear) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground">
        Select a year to view responses
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex items-center gap-4 flex-wrap">
        <Select value={levelFilter} onValueChange={setLevelFilter}>
          <SelectTrigger className="w-[150px]">
            <SelectValue placeholder="All Levels" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Levels</SelectItem>
            <SelectItem value="Grammar">Grammar</SelectItem>
            <SelectItem value="Middle">Middle</SelectItem>
            <SelectItem value="High">High</SelectItem>
            <SelectItem value="Generic">Generic</SelectItem>
          </SelectContent>
        </Select>

        <Select value={questionTypeFilter} onValueChange={setQuestionTypeFilter}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="All Question Types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Question Types</SelectItem>
            <SelectItem value="praise">Praise (Q8)</SelectItem>
            <SelectItem value="improvement">Improvement (Q9)</SelectItem>
          </SelectContent>
        </Select>

        <Select value={tagFilter} onValueChange={setTagFilter}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Filter by Tag" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Tags</SelectItem>
            {tags.map((tag) => (
              <SelectItem key={tag.name} value={tag.name}>
                {tag.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={keywordMismatchFilter} onValueChange={(v) => { setKeywordMismatchFilter(v); setPage(1) }}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="All Responses" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Responses</SelectItem>
            <SelectItem value="mismatch">Missed Keywords</SelectItem>
            <SelectItem value="no_mismatch">No Missed Keywords</SelectItem>
          </SelectContent>
        </Select>

        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger className="w-[160px]">
            <span className="flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5" />
              <SelectValue />
            </span>
          </SelectTrigger>
          <SelectContent>
            {MODEL_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          variant={sortBy === 'stability_asc' ? 'default' : 'outline'}
          size="sm"
          onClick={toggleSort}
          title={sortBy === 'stability_asc' ? 'Sorting by stability (lowest first)' : 'Sort by stability'}
        >
          <ArrowUpDown className="h-4 w-4 mr-1.5" />
          {sortBy === 'stability_asc' ? 'Stability ↑' : 'Default'}
        </Button>

        <div className="flex-1" />

        <span className="text-sm text-muted-foreground">
          {responses?.totalItems ?? 0} responses
        </span>

        <Button
          variant="outline"
          size="sm"
          onClick={handleExport}
          disabled={isExporting}
          title="Export tagging results as CSV"
        >
          {isExporting ? (
            <RefreshCw className="h-4 w-4 mr-1.5 animate-spin" />
          ) : (
            <Download className="h-4 w-4 mr-1.5" />
          )}
          Export CSV
        </Button>

        {batchTagAction}
      </div>

      {/* Table */}
      <div className="border rounded-md overflow-auto max-h-[calc(100vh-280px)]">
          <Table>
            <TableHeader className="sticky top-0 z-10 bg-background shadow-[0_1px_3px_0_rgba(0,0,0,0.1)]">
              <TableRow>
                <TableHead className="w-[140px]">Info</TableHead>
                <TableHead className="min-w-[250px]">Response</TableHead>
                {tags.map((tag) => (
                  <TableHead
                    key={tag.name}
                    className="w-[70px] text-center text-xs px-1"
                    title={tag.name}
                  >
                    {tag.name.length > 10 ? tag.name.slice(0, 10) + '...' : tag.name}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell
                    colSpan={2 + tags.length}
                    className="h-48 text-center"
                  >
                    <RefreshCw className="h-8 w-8 animate-spin mx-auto text-muted-foreground" />
                  </TableCell>
                </TableRow>
              ) : !responses?.items || responses.items.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={2 + tags.length}
                    className="h-48 text-center text-muted-foreground"
                  >
                    No responses found
                  </TableCell>
                </TableRow>
              ) : (
                responses.items.map((item) => (
                  <TaggingTableRow
                    key={item.response_id}
                    item={item}
                    tags={tags}
                    onTagToggle={onTagToggle}
                    loadingTags={loadingTags}
                    isTagging={taggingResponseIds.has(item.response_id)}
                    onLLMTag={handleLLMTag}
                  />
                ))
              )}
            </TableBody>
          </Table>
      </div>

      {/* Pagination */}
      {responses && responses.totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Rows per page:</span>
            <Select
              value={String(perPage)}
              onValueChange={(v) => {
                setPerPage(Number(v))
                setPage(1)
              }}
            >
              <SelectTrigger className="w-[80px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">10</SelectItem>
                <SelectItem value="20">20</SelectItem>
                <SelectItem value="50">50</SelectItem>
                <SelectItem value="100">100</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">
              Page {page} of {responses.totalPages}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.min(responses.totalPages, p + 1))}
                disabled={page >= responses.totalPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface TaggingTableRowProps {
  item: TaggableResponse
  tags: Array<{ name: string; keywords: string[] }>
  onTagToggle?: (responseId: string, tag: string, currentValue: boolean) => void
  loadingTags?: Set<string>
  isTagging: boolean
  onLLMTag: (responseId: string, hasOverride: boolean) => void
}

function TaggingTableRow({ item, tags, onTagToggle, loadingTags, isTagging, onLLMTag }: TaggingTableRowProps) {
  const hasKeywordMismatches = item.keyword_mismatches && item.keyword_mismatches.length > 0

  return (
    <TableRow className={item.has_override ? 'bg-blue-50/50' : undefined}>
      {/* Info column: vertical stack of metadata + AI button */}
      <TableCell className="align-top py-2">
        <div className="flex flex-col gap-1.5 text-xs">
          {/* AI tag button */}
          <div className="flex items-center gap-1.5">
            {isTagging ? (
              <RefreshCw className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            ) : (
              <button
                onClick={() => onLLMTag(item.response_id, item.has_override)}
                title="Tag with LLM"
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                <Sparkles className="h-3.5 w-3.5" />
              </button>
            )}
            <span className="font-mono text-muted-foreground">{item.respondent_id.slice(0, 8)}</span>
          </div>
          {/* Level + Type badges */}
          <div className="flex items-center gap-1">
            <Badge variant="outline" className="text-[10px] px-1 py-0 h-4">
              {item.level}
            </Badge>
            <Badge
              variant={item.question_type === 'praise' ? 'default' : 'secondary'}
              className="text-[10px] px-1 py-0 h-4"
            >
              {item.question_type === 'praise' ? 'Q8' : 'Q9'}
            </Badge>
          </div>
          {/* Stability + Flags */}
          <div className="flex items-center gap-1">
            <StabilityBadge score={item.stability_score} />
            {hasKeywordMismatches && (
              <div className="relative group inline-block">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                <div className="absolute bottom-full left-0 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10 pointer-events-none max-w-[300px]">
                  <div className="font-semibold mb-1">Missed Keywords:</div>
                  {item.keyword_mismatches.map((km, i) => (
                    <div key={i} className="text-left">
                      <span className="font-medium">{km.tag}:</span>{' '}
                      {km.keywords.join(', ')}
                    </div>
                  ))}
                  <div className="absolute top-full left-4 border-4 border-transparent border-t-gray-900" />
                </div>
              </div>
            )}
          </div>
        </div>
      </TableCell>
      {/* Response text */}
      <TableCell className="align-top py-2">
        <p className="text-sm break-words whitespace-pre-wrap">
          {item.response_text}
        </p>
      </TableCell>
      {/* Tag checkboxes */}
      {tags.map((tag) => {
        const isTagged = item.tags.includes(tag.name)
        const voteCount = item.tag_votes?.[tag.name] ?? 0
        const isLoading = loadingTags?.has(`${item.response_id}:${tag.name}`)

        return (
          <TableCell key={tag.name} className="text-center p-1 align-top">
            <TagCell
              responseId={item.response_id}
              tagName={tag.name}
              isTagged={isTagged}
              voteCount={voteCount}
              isLoading={isLoading}
              onToggle={(responseId, tagName, currentValue) => {
                if (onTagToggle) {
                  onTagToggle(responseId, tagName, currentValue)
                }
              }}
            />
          </TableCell>
        )
      })}
    </TableRow>
  )
}
