import { useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  clusteringApi,
  tagsApi,
  pipelineApi,
  type ClusterCoordinate,
} from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Play, Loader2, AlertTriangle } from 'lucide-react'
import { useToast } from '@/components/ui/use-toast'
import ClusterScatterPlot from '@/components/visualizations/ClusterScatterPlot'
import SelectedResponsesTable from '@/components/explore/SelectedResponsesTable'
import ClusterSummariesTable from '@/components/explore/ClusterSummariesTable'
import JobProgress from '@/components/jobs/JobProgress'

const FREE_RESPONSE_QUESTIONS = [
  { value: 'What makes GVCA a good choice for you and your family?', label: 'Q8 — What makes GVCA a good choice?' },
  { value: 'Please provide us with examples of how GVCA can better serve you and your family.', label: 'Q9 — How can GVCA better serve you?' },
]

export default function Explore() {
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const { selectedYear } = useAppStore()

  // Filters
  const [filterLevel, setFilterLevel] = useState<string>('all')
  const [filterQuestion, setFilterQuestion] = useState<string>('all')
  const [filterTag, setFilterTag] = useState<string>('all')

  // Selection state
  const [selectedResponses, setSelectedResponses] = useState<ClusterCoordinate[]>([])

  // Active clustering job tracking
  const [clusteringJobId, setClusteringJobId] = useState<string | null>(null)

  const handleClusteringComplete = useCallback(() => {
    // Job finished — invalidate queries to show fresh data
    queryClient.invalidateQueries({ queryKey: ['cluster-coordinates', selectedYear] })
    queryClient.invalidateQueries({ queryKey: ['cluster-summaries', selectedYear] })
    queryClient.invalidateQueries({ queryKey: ['cluster-metadata', selectedYear] })
    queryClient.invalidateQueries({ queryKey: ['pipeline-status', selectedYear] })
    setClusteringJobId(null)
  }, [queryClient, selectedYear])

  // Fetch pipeline status
  const { data: pipelineStatus } = useQuery({
    queryKey: ['pipeline-status', selectedYear],
    queryFn: () => selectedYear ? pipelineApi.getStatus(selectedYear) : null,
    enabled: !!selectedYear,
  })

  // Fetch coordinates with tags
  const { data: coordinatesData, isLoading: isLoadingCoords } = useQuery({
    queryKey: ['cluster-coordinates', selectedYear, 'with-tags'],
    queryFn: () => selectedYear ? clusteringApi.getCoordinates(selectedYear, { includeTags: true }) : null,
    enabled: !!selectedYear,
  })

  // Fetch cluster summaries
  const { data: summariesData } = useQuery({
    queryKey: ['cluster-summaries', selectedYear],
    queryFn: () => selectedYear ? clusteringApi.getSummaries(selectedYear) : null,
    enabled: !!selectedYear,
  })

  // Fetch cluster metadata
  const { data: metadataData } = useQuery({
    queryKey: ['cluster-metadata', selectedYear],
    queryFn: () => selectedYear ? clusteringApi.getMetadata(selectedYear) : null,
    enabled: !!selectedYear,
  })

  // Fetch taxonomy
  const { data: taxonomy } = useQuery({
    queryKey: ['taxonomy'],
    queryFn: () => tagsApi.getTaxonomy(),
  })

  // Run clustering mutation
  const clusteringMutation = useMutation({
    mutationFn: () => clusteringApi.start(selectedYear!),
    onSuccess: (data) => {
      setClusteringJobId(data.job_id)
      useAppStore.getState().addJob({
        id: data.job_id,
        job_type: 'clustering',
        year: selectedYear!,
        status: 'running',
        progress: 0,
        total_items: 0,
        processed_items: 0,
        error_message: null,
        started_at: new Date().toISOString(),
        completed_at: null,
        metadata: {},
      })
      toast({ title: 'Clustering started' })
    },
    onError: (err) => {
      toast({
        title: 'Failed to start clustering',
        description: err instanceof Error ? err.message : 'Unknown error',
        variant: 'destructive',
      })
    },
  })

  const handlePointClick = useCallback((point: ClusterCoordinate) => {
    setSelectedResponses([point])
  }, [])

  const handleSelectionChange = useCallback((points: ClusterCoordinate[]) => {
    setSelectedResponses(points)
  }, [])

  // No year selected
  if (!selectedYear) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Card className="w-96">
          <CardHeader>
            <CardTitle>No Year Selected</CardTitle>
            <CardDescription>Select a survey year from the dropdown to explore clusters.</CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  const hasClusteringData =
    coordinatesData?.coordinates && coordinatesData.coordinates.length > 0

  // No clustering data — show init panel
  if (!isLoadingCoords && !hasClusteringData) {
    return (
      <div className="space-y-6">
        <h2 className="text-3xl font-bold tracking-tight">Explore</h2>
        {clusteringJobId && (
          <JobProgress jobId={clusteringJobId} onComplete={handleClusteringComplete} />
        )}
        {!clusteringJobId && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-16 gap-4">
              {!pipelineStatus?.loaded ? (
                <p className="text-muted-foreground">Load data using the toolbar, then run clustering.</p>
              ) : pipelineStatus?.embeddings_complete ? (
                <p className="text-muted-foreground">Embeddings are ready. Run clustering to generate the scatter plot.</p>
              ) : (
                <p className="text-muted-foreground">No clustering data available. Run clustering to generate embeddings and clusters.</p>
              )}
              {pipelineStatus?.loaded && (
                <Button
                  onClick={() => clusteringMutation.mutate()}
                  disabled={clusteringMutation.isPending}
                >
                  {clusteringMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Run Clustering
                </Button>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  const coordinates = coordinatesData?.coordinates ?? []
  const summaries = summariesData?.clusters ?? []
  const metadata = metadataData?.metadata ?? []
  const tagList = taxonomy?.tags ?? []

  // Apply filters to coordinates for use by both scatter plot and summaries
  const filteredCoordinates = coordinates.filter((c) => {
    if (filterLevel !== 'all' && c.level !== filterLevel) return false
    if (filterQuestion !== 'all' && c.question !== filterQuestion) return false
    if (filterTag !== 'all' && !(c.tags ?? []).includes(filterTag)) return false
    return true
  })
  const hasActiveFilters = filterLevel !== 'all' || filterQuestion !== 'all' || filterTag !== 'all'

  // Staleness detection: if many points are missing response_text, the data is stale
  const enrichedCount = coordinates.filter(c => c.response_text).length
  const isDataStale = coordinates.length > 0 && enrichedCount < coordinates.length * 0.5
  const missingCount = coordinates.length - enrichedCount

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold tracking-tight">Explore</h2>

      {/* Clustering job progress */}
      {clusteringJobId && (
        <JobProgress jobId={clusteringJobId} onComplete={handleClusteringComplete} />
      )}

      {/* Stale data warning */}
      {isDataStale && !clusteringJobId && (
        <Card className="border-amber-300 bg-amber-50">
          <CardContent className="flex items-center justify-between py-3">
            <div className="flex items-center gap-2 text-sm text-amber-800">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>
                Stale data detected: response text is missing for {missingCount} of {coordinates.length} points. Re-run clustering to fix.
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => clusteringMutation.mutate()}
              disabled={clusteringMutation.isPending}
            >
              {clusteringMutation.isPending ? (
                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
              ) : (
                <Play className="h-3 w-3 mr-1" />
              )}
              Re-run Clustering
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Filters */}
      <Card>
        <CardContent className="flex gap-4 py-3">
          <Select value={filterLevel} onValueChange={setFilterLevel}>
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="All Levels" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="Grammar">Grammar</SelectItem>
              <SelectItem value="Middle">Middle</SelectItem>
              <SelectItem value="High">High</SelectItem>
            </SelectContent>
          </Select>

          <Select value={filterQuestion} onValueChange={setFilterQuestion}>
            <SelectTrigger className="w-[320px]">
              <SelectValue placeholder="All Questions" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Questions</SelectItem>
              {FREE_RESPONSE_QUESTIONS.map((q) => (
                <SelectItem key={q.value} value={q.value}>
                  {q.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={filterTag} onValueChange={setFilterTag}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="All Tags" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Tags</SelectItem>
              {tagList.map((tag) => (
                <SelectItem key={tag.name} value={tag.name}>
                  {tag.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {(filterLevel !== 'all' || filterQuestion !== 'all' || filterTag !== 'all') && (
            <button
              className="text-sm text-muted-foreground hover:text-foreground"
              onClick={() => {
                setFilterLevel('all')
                setFilterQuestion('all')
                setFilterTag('all')
              }}
            >
              Clear filters
            </button>
          )}
        </CardContent>
      </Card>

      <Tabs defaultValue="visualization">
        <TabsList>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="summaries">Cluster Summaries</TabsTrigger>
        </TabsList>

        {/* Visualization Tab */}
        <TabsContent value="visualization" className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Response Clusters</CardTitle>
              <CardDescription>
                Click a point to view its response, or use lasso select for multiple responses.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingCoords ? (
                <div className="h-[500px] flex items-center justify-center text-muted-foreground">
                  Loading cluster data...
                </div>
              ) : (
                <ClusterScatterPlot
                  data={coordinates}
                  width={900}
                  height={500}
                  onPointClick={handlePointClick}
                  onSelectionChange={handleSelectionChange}
                  filterLevel={filterLevel === 'all' ? undefined : filterLevel}
                  filterTag={filterTag === 'all' ? undefined : filterTag}
                  filterQuestion={filterQuestion === 'all' ? undefined : filterQuestion}
                  hideLegend
                />
              )}
            </CardContent>
          </Card>

          {selectedResponses.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">
                  Selected Responses ({selectedResponses.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <SelectedResponsesTable responses={selectedResponses} />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Cluster Summaries Tab */}
        <TabsContent value="summaries" className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Cluster Summaries</CardTitle>
              <CardDescription>
                {summaries.length} clusters. Click a name or description to edit. Expand a row to see responses and edit tags.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {summaries.length === 0 ? (
                <p className="text-muted-foreground py-8 text-center">
                  No cluster summaries available.
                </p>
              ) : (
                <ClusterSummariesTable
                  year={selectedYear}
                  summaries={summaries}
                  metadata={metadata}
                  coordinates={hasActiveFilters ? filteredCoordinates : coordinates}
                  taxonomy={tagList}
                  filtered={hasActiveFilters}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
