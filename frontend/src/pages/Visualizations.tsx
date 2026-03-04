import { useQueries, useQuery } from '@tanstack/react-query'
import { dataApi, pipelineApi, taggingApi } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, Info } from 'lucide-react'
import QuestionPanel from '@/components/visualizations/QuestionPanel'
import type { QuestionPanelData, SubgroupBar, TrendBar } from '@/components/visualizations/QuestionPanel'
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

/**
 * Group question_totals rows by Question text.
 * Returns groups in the order questions appear (Q1-Q7 scaled only).
 */
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

/**
 * Compute weighted average from a 4-element percentage array.
 * [%_top, %_second, %_third, %_bottom] where top=4, bottom=1.
 */
function computeWeightedAverage(distribution: number[]): number {
  if (distribution.length !== 4) return 0
  const sum = distribution.reduce((acc, pct, i) => acc + (4 - i) * pct, 0)
  return sum / 100
}

/**
 * Extract the distribution percentages for a specific level from the rows of a single question.
 * Returns [%_top, %_second, %_third, %_bottom] in the order the scale defines them.
 */
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

/**
 * Extract the total distribution percentages from the rows of a single question.
 */
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

/**
 * Build QuestionPanelData for a single question across all loaded years.
 */
function buildQuestionPanelData(
  questionIndex: number,
  allYearsData: Record<string, QuestionTotalRow[][]>,
  selectedYear: string,
  availableYears: string[]
): QuestionPanelData | null {
  const survey = SURVEY_QUESTIONS[questionIndex]
  const scale = QUESTION_SCALES[questionIndex]
  if (!survey || !scale) return null

  // Get rows for the selected year
  const selectedYearGroups = allYearsData[selectedYear]
  if (!selectedYearGroups || !selectedYearGroups[questionIndex]) return null

  const selectedRows = selectedYearGroups[questionIndex]

  // Build subgroup bars (Grammar, Middle, High) for selected year
  const subgroups: SubgroupBar[] = SCHOOL_LEVELS.map((level) => {
    const dist = extractLevelDistribution(selectedRows, level, scale.labels)
    return {
      level,
      distribution: dist,
      weightedAverage: computeWeightedAverage(dist),
    }
  })

  // Build trend bars for each available year
  const sortedYears = [...availableYears].sort()
  const trends: TrendBar[] = sortedYears
    .filter((y) => allYearsData[y]?.[questionIndex])
    .map((year) => {
      const rows = allYearsData[year][questionIndex]
      const dist = extractTotalDistribution(rows, scale.labels)
      return {
        year,
        distribution: dist,
        weightedAverage: computeWeightedAverage(dist),
      }
    })

  // Build historical averages: year -> level -> avg
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

  for (const item of goodChoice ?? []) {
    map.set(item.tag, {
      tag: item.tag,
      goodChoiceCount: item.count,
      betterServeCount: 0,
      totalCount: item.count,
    })
  }

  for (const item of betterServe ?? []) {
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

// ============ Page Component ============

export default function Visualizations() {
  const { selectedYear } = useAppStore()

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
      staleTime: 5 * 60 * 1000,
    })),
  })

  // Fetch tag distributions (Q8 = good choice, Q9 = better serve)
  const goodChoiceQuery = useQuery({
    queryKey: ['tag-distribution', selectedYear, 'good choice'],
    queryFn: () => taggingApi.getDistribution(selectedYear!, undefined, 'good choice'),
    enabled: !!selectedYear,
  })

  const betterServeQuery = useQuery({
    queryKey: ['tag-distribution', selectedYear, 'better serve'],
    queryFn: () => taggingApi.getDistribution(selectedYear!, undefined, 'better serve'),
    enabled: !!selectedYear,
  })

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

  // Check loading state
  const isLoadingStats = statisticsQueries.some((q) => q.isLoading)
  const statsErrors = statisticsQueries.filter((q) => q.isError)

  if (isLoadingStats) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  // Build allYearsData: year -> array of question groups (index 0-6)
  const allYearsData: Record<string, QuestionTotalRow[][]> = {}
  for (let i = 0; i < availableYears.length; i++) {
    const year = availableYears[i]
    const result = statisticsQueries[i]?.data
    if (result?.question_totals) {
      const groups = groupQuestionTotals(result.question_totals as QuestionTotalRow[])
      // Convert map to array in order (first 7 groups = Q1-Q7)
      allYearsData[year] = Array.from(groups.values()).slice(0, 7)
    }
  }

  // Build panel data for each of the 7 questions
  const panels: QuestionPanelData[] = []
  for (let i = 0; i < 7; i++) {
    const panel = buildQuestionPanelData(i, allYearsData, selectedYear, availableYears)
    if (panel) panels.push(panel)
  }

  // Merge tag frequency data
  const tagFrequencyData = mergeTagDistributions(
    goodChoiceQuery.data?.distribution,
    betterServeQuery.data?.distribution
  )
  const isTaggingLoading = goodChoiceQuery.isLoading || betterServeQuery.isLoading
  const hasTaggingData = (goodChoiceQuery.data?.total_tags ?? 0) > 0 || (betterServeQuery.data?.total_tags ?? 0) > 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Visualizations</h2>
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
        <div className="space-y-4">
          {panels.map((panel) => (
            <ExportableCard
              key={panel.questionIndex}
              filename={`Q${panel.presentationNumber}-${panel.shortTitle.replace(/\s+/g, '-')}-${selectedYear}`}
              contentClassName="pt-4 pb-3"
            >
              <QuestionPanel data={panel} selectedYear={selectedYear} />
            </ExportableCard>
          ))}
        </div>
      )}

      {/* Tag Frequency Diverging Bar Chart */}
      <ExportableCard
        title="Open Response Tag Frequency"
        filename={`Tag-Frequency-${selectedYear}`}
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
    </div>
  )
}
