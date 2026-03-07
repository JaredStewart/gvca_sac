import { useMemo, useState } from 'react'
import { useQueries, useQuery, useMutation } from '@tanstack/react-query'
import { dataApi, pipelineApi, taggingApi, chartsApi } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, Info, Download, Check } from 'lucide-react'
import SubgroupChart, { getSubgroupCsvData, getSubgroupCsvColumns } from '@/components/visualizations/SubgroupChart'
import type { SubgroupBar } from '@/components/visualizations/SubgroupChart'
import YearOverYearChart, { getTrendCsvData, getTrendCsvColumns } from '@/components/visualizations/YearOverYearChart'
import type { TrendBar } from '@/components/visualizations/YearOverYearChart'
import CompositeLineChart, { getCompositeCsvData, compositeCsvColumns } from '@/components/visualizations/CompositeLineChart'
import type { CompositeDataPoint } from '@/components/visualizations/CompositeLineChart'
import GoodChoiceBetterServeChart, { getGoodChoiceBetterServeCsvData, goodChoiceBetterServeCsvColumns } from '@/components/visualizations/GoodChoiceBetterServeChart'
import type { FreeResponseCounts } from '@/components/visualizations/GoodChoiceBetterServeChart'
import TagTrendChart, { getTagTrendCsvData, getTagTrendCsvColumns } from '@/components/visualizations/TagTrendChart'
import type { TagTrendDataPoint } from '@/components/visualizations/TagTrendChart'
import TagFrequencyChart from '@/components/visualizations/TagFrequencyChart'
import ExportableCard from '@/components/visualizations/ExportableCard'
import { SURVEY_QUESTIONS, QUESTION_SCALES, SCHOOL_LEVELS } from '@/constants'

// ============ Data Transformation Utilities ============

interface QuestionTotalRow {
  Question: string
  Response: string
  N_total: number
  '%_total': number
  N_Grammar: number
  '%_Grammar': number
  N_Middle: number
  '%_Middle': number
  N_High: number
  '%_High': number
  [key: string]: unknown
}

interface QuestionPanelData {
  questionIndex: number
  presentationNumber: number
  shortTitle: string
  questionText: string
  scaleLabels: string[]
  subgroups: SubgroupBar[]
  trends: TrendBar[]
  historicalAverages: Record<string, Record<string, number>>
}

function groupQuestionTotals(questionTotals: QuestionTotalRow[]): Map<string, QuestionTotalRow[]> {
  const groups = new Map<string, QuestionTotalRow[]>()
  for (const row of questionTotals) {
    const q = row.Question
    if (!groups.has(q)) {
      groups.set(q, [])
    }
    groups.get(q)!.push(row)
  }
  return groups
}

function computeWeightedAverage(distribution: number[]): number {
  if (distribution.length !== 4) return 0
  const sum = distribution.reduce((acc, pct, i) => acc + (4 - i) * pct, 0)
  return sum / 100
}

function extractLevelDistribution(
  rows: QuestionTotalRow[],
  level: string,
  scaleLabels: string[]
): number[] {
  return scaleLabels.map((label) => {
    const row = rows.find((r) => r.Response === label)
    if (!row) return 0
    const key = `%_${level}` as keyof QuestionTotalRow
    const val = row[key]
    return typeof val === 'number' ? val : 0
  })
}

function extractLevelCounts(
  rows: QuestionTotalRow[],
  level: string,
  scaleLabels: string[]
): number[] {
  return scaleLabels.map((label) => {
    const row = rows.find((r) => r.Response === label)
    if (!row) return 0
    const key = `N_${level}` as keyof QuestionTotalRow
    const val = row[key]
    return typeof val === 'number' ? val : 0
  })
}

function extractTotalDistribution(
  rows: QuestionTotalRow[],
  scaleLabels: string[]
): number[] {
  return scaleLabels.map((label) => {
    const row = rows.find((r) => r.Response === label)
    if (!row) return 0
    return row['%_total'] ?? 0
  })
}

function extractTotalCounts(
  rows: QuestionTotalRow[],
  scaleLabels: string[]
): number[] {
  return scaleLabels.map((label) => {
    const row = rows.find((r) => r.Response === label)
    if (!row) return 0
    return row['N_total'] ?? 0
  })
}

function buildQuestionPanelData(
  questionIndex: number,
  allYearsData: Record<string, QuestionTotalRow[][]>,
  selectedYear: string,
  availableYears: string[]
): QuestionPanelData | null {
  const survey = SURVEY_QUESTIONS[questionIndex]
  const scale = QUESTION_SCALES[questionIndex]
  if (!survey || !scale) return null

  const selectedYearGroups = allYearsData[selectedYear]
  if (!selectedYearGroups || !selectedYearGroups[questionIndex]) return null

  const selectedRows = selectedYearGroups[questionIndex]

  const subgroups: SubgroupBar[] = SCHOOL_LEVELS.map((level) => {
    const dist = extractLevelDistribution(selectedRows, level, scale.labels)
    const counts = extractLevelCounts(selectedRows, level, scale.labels)
    return {
      level,
      distribution: dist,
      counts,
      weightedAverage: computeWeightedAverage(dist),
    }
  })

  const sortedYears = [...availableYears].sort()
  const trends: TrendBar[] = sortedYears
    .filter((y) => allYearsData[y]?.[questionIndex])
    .map((year) => {
      const rows = allYearsData[year][questionIndex]
      const dist = extractTotalDistribution(rows, scale.labels)
      const counts = extractTotalCounts(rows, scale.labels)
      return {
        year,
        distribution: dist,
        counts,
        weightedAverage: computeWeightedAverage(dist),
      }
    })

  const historicalAverages: Record<string, Record<string, number>> = {}
  for (const year of sortedYears) {
    if (!allYearsData[year]?.[questionIndex]) continue
    const rows = allYearsData[year][questionIndex]
    historicalAverages[year] = {}
    for (const level of SCHOOL_LEVELS) {
      const dist = extractLevelDistribution(rows, level, scale.labels)
      historicalAverages[year][level] = computeWeightedAverage(dist)
    }
  }

  return {
    questionIndex,
    presentationNumber: survey.presentationNumber,
    shortTitle: survey.shortTitle,
    questionText: survey.fullText,
    scaleLabels: [...scale.labels],
    subgroups,
    trends,
    historicalAverages,
  }
}

// ============ Composite Satisfaction Data ============

function buildCompositeData(
  allYearsData: Record<string, QuestionTotalRow[][]>,
  availableYears: string[]
): CompositeDataPoint[] {
  const sortedYears = [...availableYears].sort()

  return sortedYears
    .filter((year) => allYearsData[year])
    .map((year) => {
      const yearGroups = allYearsData[year]
      // Sum counts by tier position across all questions
      const tierCounts = [0, 0, 0, 0]
      let totalCount = 0

      for (let qi = 0; qi < 7; qi++) {
        const rows = yearGroups[qi]
        if (!rows) continue
        const scale = QUESTION_SCALES[qi]
        if (!scale) continue

        // Use total counts across all levels
        const counts = extractTotalCounts(rows, scale.labels)
        for (let ti = 0; ti < 4; ti++) {
          tierCounts[ti] += counts[ti]
          totalCount += counts[ti]
        }
      }

      if (totalCount === 0) {
        return { year, extremelySatisfied: 0, satisfied: 0, somewhatSatisfied: 0, dissatisfied: 0 }
      }

      return {
        year,
        extremelySatisfied: Math.round(tierCounts[0] / totalCount * 1000) / 10,
        satisfied: Math.round(tierCounts[1] / totalCount * 1000) / 10,
        somewhatSatisfied: Math.round(tierCounts[2] / totalCount * 1000) / 10,
        dissatisfied: Math.round(tierCounts[3] / totalCount * 1000) / 10,
      }
    })
}

// Opposite-condition marker tags — excluded from frequency/trend charts
const EXCLUDED_TAGS = new Set(['Concern', 'No improvement listed'])

// ============ Tag Frequency Types & Merge ============

export interface TagFrequencyData {
  tag: string
  goodChoiceCount: number
  betterServeCount: number
  totalCount: number
}

function mergeTagDistributions(
  goodChoice: Array<{ tag: string; count: number }> | undefined,
  betterServe: Array<{ tag: string; count: number }> | undefined
): TagFrequencyData[] {
  const map = new Map<string, TagFrequencyData>()

  for (const item of (goodChoice ?? []).filter((i) => !EXCLUDED_TAGS.has(i.tag))) {
    map.set(item.tag, {
      tag: item.tag,
      goodChoiceCount: item.count,
      betterServeCount: 0,
      totalCount: item.count,
    })
  }

  for (const item of (betterServe ?? []).filter((i) => !EXCLUDED_TAGS.has(i.tag))) {
    const existing = map.get(item.tag)
    if (existing) {
      existing.betterServeCount = item.count
      existing.totalCount = existing.goodChoiceCount + item.count
    } else {
      map.set(item.tag, {
        tag: item.tag,
        goodChoiceCount: 0,
        betterServeCount: item.count,
        totalCount: item.count,
      })
    }
  }

  return Array.from(map.values()).sort((a, b) => b.goodChoiceCount - a.goodChoiceCount)
}

// ============ Tag Trend Data Builder ============

function buildTagTrendData(
  allYearsGoodChoice: Record<string, Array<{ tag: string; count: number }> | undefined>,
  allYearsBetterServe: Record<string, Array<{ tag: string; count: number }> | undefined>,
  availableYears: string[]
): { data: TagTrendDataPoint[]; goodChoiceTags: string[]; betterServeTags: string[] } {
  const sortedYears = [...availableYears].sort()

  // Collect all unique tags
  const gcTagSet = new Set<string>()
  const bsTagSet = new Set<string>()

  for (const year of sortedYears) {
    for (const item of allYearsGoodChoice[year] ?? []) {
      if (!EXCLUDED_TAGS.has(item.tag)) gcTagSet.add(item.tag)
    }
    for (const item of allYearsBetterServe[year] ?? []) {
      if (!EXCLUDED_TAGS.has(item.tag)) bsTagSet.add(item.tag)
    }
  }

  const goodChoiceTags = Array.from(gcTagSet).sort()
  const betterServeTags = Array.from(bsTagSet).sort()

  const data: TagTrendDataPoint[] = sortedYears.map((year) => {
    const gcDist = (allYearsGoodChoice[year] ?? []).filter((d) => !EXCLUDED_TAGS.has(d.tag))
    const bsDist = (allYearsBetterServe[year] ?? []).filter((d) => !EXCLUDED_TAGS.has(d.tag))

    const gcMap = new Map(gcDist.map((d) => [d.tag, d.count]))
    const bsMap = new Map(bsDist.map((d) => [d.tag, d.count]))

    const point: TagTrendDataPoint = {
      year,
      goodChoiceTotal: gcDist.reduce((sum, d) => sum + d.count, 0),
      betterServeTotal: bsDist.reduce((sum, d) => sum + d.count, 0),
    }

    for (const tag of goodChoiceTags) {
      point[`goodChoice_${tag}`] = gcMap.get(tag) ?? 0
    }
    for (const tag of betterServeTags) {
      point[`betterServe_${tag}`] = -(bsMap.get(tag) ?? 0)
    }

    return point
  })

  return { data, goodChoiceTags, betterServeTags }
}

// ============ Page Component ============

export default function Visualizations() {
  const { selectedYear } = useAppStore()
  const [exportStatus, setExportStatus] = useState<'idle' | 'exporting' | 'done'>('idle')

  // Fetch available years
  const { data: yearsData } = useQuery({
    queryKey: ['years'],
    queryFn: () => pipelineApi.getYears(),
  })

  const availableYears = yearsData?.years ?? []

  // Fetch statistics for ALL available years in parallel
  const statisticsQueries = useQueries({
    queries: availableYears.map((year) => ({
      queryKey: ['statistics', year],
      queryFn: () => dataApi.getStatistics(year),
      staleTime: Infinity,
    })),
  })

  // Fetch tag distributions for selected year
  const goodChoiceQuery = useQuery({
    queryKey: ['tag-distribution', selectedYear, 'good choice'],
    queryFn: () => taggingApi.getDistribution(selectedYear!, undefined, 'good choice'),
    enabled: !!selectedYear,
    staleTime: Infinity,
  })

  const betterServeQuery = useQuery({
    queryKey: ['tag-distribution', selectedYear, 'better serve'],
    queryFn: () => taggingApi.getDistribution(selectedYear!, undefined, 'better serve'),
    enabled: !!selectedYear,
    staleTime: Infinity,
  })

  // Fetch tag distributions for ALL years (for tag trend chart)
  const allYearsGcQueries = useQueries({
    queries: availableYears.map((year) => ({
      queryKey: ['tag-distribution', year, 'good choice'],
      queryFn: () => taggingApi.getDistribution(year, undefined, 'good choice'),
      staleTime: Infinity,
    })),
  })

  const allYearsBsQueries = useQueries({
    queries: availableYears.map((year) => ({
      queryKey: ['tag-distribution', year, 'better serve'],
      queryFn: () => taggingApi.getDistribution(year, undefined, 'better serve'),
      staleTime: Infinity,
    })),
  })

  // Fetch free response counts for Good Choice / Better Serve chart
  const freeResponseCountsQuery = useQuery({
    queryKey: ['free-response-counts', selectedYear],
    queryFn: () => dataApi.getFreeResponseCounts(selectedYear!),
    enabled: !!selectedYear,
    staleTime: Infinity,
  })

  // Export All mutation
  const exportAllMutation = useMutation({
    mutationFn: () => chartsApi.exportAll(selectedYear!, availableYears),
    onSuccess: () => {
      setExportStatus('done')
      setTimeout(() => setExportStatus('idle'), 3000)
    },
    onError: () => setExportStatus('idle'),
  })

  // ============ Memoized Data (must be before any early returns) ============

  const isLoadingStats = statisticsQueries.some((q) => q.isLoading)
  const statsErrors = statisticsQueries.filter((q) => q.isError)

  const allYearsData = useMemo(() => {
    const result: Record<string, QuestionTotalRow[][]> = {}
    for (let i = 0; i < availableYears.length; i++) {
      const year = availableYears[i]
      const data = statisticsQueries[i]?.data
      if (data?.question_totals) {
        const groups = groupQuestionTotals(data.question_totals as QuestionTotalRow[])
        result[year] = Array.from(groups.values()).slice(0, 7)
      }
    }
    return result
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableYears.join(','), statisticsQueries.map((q) => q.dataUpdatedAt).join(',')])

  const panels = useMemo(() => {
    if (!selectedYear) return []
    const result: QuestionPanelData[] = []
    for (let i = 0; i < 7; i++) {
      const panel = buildQuestionPanelData(i, allYearsData, selectedYear, availableYears)
      if (panel) result.push(panel)
    }
    return result
  }, [allYearsData, selectedYear, availableYears])

  const compositeData = useMemo(
    () => buildCompositeData(allYearsData, availableYears),
    [allYearsData, availableYears]
  )

  const tagFrequencyData = useMemo(
    () => mergeTagDistributions(goodChoiceQuery.data?.distribution, betterServeQuery.data?.distribution),
    [goodChoiceQuery.data, betterServeQuery.data]
  )

  const tagTrendData = useMemo(() => {
    const allGc: Record<string, Array<{ tag: string; count: number }> | undefined> = {}
    const allBs: Record<string, Array<{ tag: string; count: number }> | undefined> = {}
    for (let i = 0; i < availableYears.length; i++) {
      const year = availableYears[i]
      allGc[year] = allYearsGcQueries[i]?.data?.distribution
      allBs[year] = allYearsBsQueries[i]?.data?.distribution
    }
    return buildTagTrendData(allGc, allBs, availableYears)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    availableYears.join(','),
    allYearsGcQueries.map((q) => q.dataUpdatedAt).join(','),
    allYearsBsQueries.map((q) => q.dataUpdatedAt).join(','),
  ])

  const isTaggingLoading = goodChoiceQuery.isLoading || betterServeQuery.isLoading
  const hasTaggingData = (goodChoiceQuery.data?.total_tags ?? 0) > 0 || (betterServeQuery.data?.total_tags ?? 0) > 0
  const freeResponseCounts = freeResponseCountsQuery.data as FreeResponseCounts | undefined

  const hasTagTrendData = tagTrendData.data.length >= 2 &&
    tagTrendData.data.some((d) => d.goodChoiceTotal > 0 || d.betterServeTotal > 0)

  // ============ Early Returns (after all hooks) ============

  if (!selectedYear) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Card className="w-96">
          <CardHeader>
            <CardTitle>No Year Selected</CardTitle>
          </CardHeader>
        </Card>
      </div>
    )
  }

  if (isLoadingStats) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Visualizations</h2>
        <button
          onClick={() => {
            setExportStatus('exporting')
            exportAllMutation.mutate()
          }}
          disabled={exportStatus !== 'idle'}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {exportStatus === 'exporting' ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : exportStatus === 'done' ? (
            <Check className="h-4 w-4" />
          ) : (
            <Download className="h-4 w-4" />
          )}
          {exportStatus === 'exporting' ? 'Exporting...' : exportStatus === 'done' ? 'Exported to Artifacts' : 'Export All to Artifacts'}
        </button>
      </div>

      {statsErrors.length > 0 && (
        <Alert variant="destructive">
          <AlertDescription>
            Failed to load statistics for some years. Charts may be incomplete.
          </AlertDescription>
        </Alert>
      )}

      {panels.length === 0 ? (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            No statistics data available for {selectedYear}. Initialize the pipeline from the Dashboard first.
          </AlertDescription>
        </Alert>
      ) : (
        <>
          {/* Overall Composite Satisfaction Scores */}
          {compositeData.length > 0 && (
            <ExportableCard
              title="Overall Composite Satisfaction Scores"
              filename={`Overall-Composite-Satisfaction`}
              csvData={getCompositeCsvData(compositeData)}
              csvColumns={compositeCsvColumns}
              csvFilename="Overall-Composite-Satisfaction"
            >
              <CompositeLineChart data={compositeData} />
            </ExportableCard>
          )}

          {/* Per-Question Charts - Subgroup and YoY separately */}
          <div className="space-y-4">
            {panels.map((panel) => (
              <div key={panel.questionIndex} className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <ExportableCard
                  title={`Q${panel.presentationNumber}: ${panel.shortTitle} - ${selectedYear} Subgroup Responses`}
                  filename={`Q${panel.presentationNumber}-Subgroups-${selectedYear}`}
                  csvData={getSubgroupCsvData(panel.subgroups, panel.scaleLabels)}
                  csvColumns={getSubgroupCsvColumns(panel.scaleLabels)}
                  csvFilename={`Q${panel.presentationNumber}-Subgroups-${selectedYear}`}
                  contentClassName="pt-4 pb-3"
                >
                  <SubgroupChart
                    presentationNumber={panel.presentationNumber}
                    shortTitle={panel.shortTitle}
                    scaleLabels={panel.scaleLabels}
                    subgroups={panel.subgroups}
                    selectedYear={selectedYear}
                    historicalAverages={panel.historicalAverages}
                  />
                </ExportableCard>
                <ExportableCard
                  title={`Q${panel.presentationNumber}: ${panel.shortTitle} - Year-Over-Year Trend`}
                  filename={`Q${panel.presentationNumber}-YoY-Trend`}
                  csvData={getTrendCsvData(panel.trends, panel.scaleLabels)}
                  csvColumns={getTrendCsvColumns(panel.scaleLabels)}
                  csvFilename={`Q${panel.presentationNumber}-YoY-Trend`}
                  contentClassName="pt-4 pb-3"
                >
                  <YearOverYearChart
                    presentationNumber={panel.presentationNumber}
                    shortTitle={panel.shortTitle}
                    scaleLabels={panel.scaleLabels}
                    trends={panel.trends}
                  />
                </ExportableCard>
              </div>
            ))}
          </div>

          {/* Tag Frequency Diverging Bar Chart */}
          <ExportableCard
            title="Open Response Tag Frequency"
            filename={`Tag-Frequency-${selectedYear}`}
            csvData={tagFrequencyData.map((d) => ({ Tag: d.tag, 'Good Choice Count': d.goodChoiceCount, 'Better Serve Count': d.betterServeCount, 'Total Count': d.totalCount }))}
            csvColumns={[
              { header: 'Tag', accessor: (r) => r['Tag'] as string },
              { header: 'Good Choice Count', accessor: (r) => r['Good Choice Count'] as number },
              { header: 'Better Serve Count', accessor: (r) => r['Better Serve Count'] as number },
              { header: 'Total Count', accessor: (r) => r['Total Count'] as number },
            ]}
            csvFilename={`Tag-Frequency-${selectedYear}`}
          >
            {isTaggingLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : !hasTaggingData ? (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  Tagging has not been run for {selectedYear}. Run tagging from the Tagging page first.
                </AlertDescription>
              </Alert>
            ) : (
              <TagFrequencyChart data={tagFrequencyData} />
            )}
          </ExportableCard>

          {/* Good Choice / Better Serve Difference Chart */}
          {freeResponseCounts && freeResponseCounts.total_good_choice > 0 && (
            <ExportableCard
              title="Good Choice vs Better Serve Responses"
              filename={`Good-Choice-Better-Serve-${selectedYear}`}
              csvData={getGoodChoiceBetterServeCsvData(freeResponseCounts)}
              csvColumns={goodChoiceBetterServeCsvColumns}
              csvFilename={`Good-Choice-Better-Serve-${selectedYear}`}
            >
              <GoodChoiceBetterServeChart data={freeResponseCounts} />
            </ExportableCard>
          )}

          {/* Tag Response Trend (YoY) */}
          {hasTagTrendData && (
            <ExportableCard
              title="Tag Response Trend (Good Choice / Better Serve by Year)"
              filename={`Tag-Response-Trend`}
              csvData={getTagTrendCsvData(tagTrendData.data, tagTrendData.goodChoiceTags, tagTrendData.betterServeTags)}
              csvColumns={getTagTrendCsvColumns(tagTrendData.goodChoiceTags, tagTrendData.betterServeTags)}
              csvFilename="Tag-Response-Trend"
            >
              <TagTrendChart
                data={tagTrendData.data}
                goodChoiceTags={tagTrendData.goodChoiceTags}
                betterServeTags={tagTrendData.betterServeTags}
              />
            </ExportableCard>
          )}
        </>
      )}
    </div>
  )
}
