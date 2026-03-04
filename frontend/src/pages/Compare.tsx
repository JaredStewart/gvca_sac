import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { pipelineApi, dataApi, chartsApi } from '@/api/client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar,
} from 'recharts'
import { BarChart3, Loader2 } from 'lucide-react'
import { CHART_COLORS } from '@/constants'

const QUESTIONS_SHORT = [
  { key: 'Q1', label: 'Education Satisfaction' },
  { key: 'Q2', label: 'Intellectual Growth' },
  { key: 'Q3', label: 'Core Virtues' },
  { key: 'Q4', label: 'Character Growth' },
  { key: 'Q5', label: 'Teacher Communication' },
  { key: 'Q6', label: 'Leadership Communication' },
  { key: 'Q7', label: 'Welcoming Community' },
]

export default function Compare() {
  const [selectedYears, setSelectedYears] = useState<string[]>([])

  // Fetch available years
  const { data: yearsData } = useQuery({
    queryKey: ['years'],
    queryFn: () => pipelineApi.getYears(),
  })

  // Fetch statistics for selected years
  const statsQueries = useQuery({
    queryKey: ['compare-stats', selectedYears],
    queryFn: async () => {
      const results = await Promise.allSettled(
        selectedYears.map(async (year) => {
          // Try to get stats, init if not loaded
          await pipelineApi.init(year)
          const stats = await dataApi.getStatistics(year)
          return { year, stats }
        })
      )
      // Extract successful results
      return results
        .filter((r): r is PromiseFulfilledResult<{ year: string; stats: Awaited<ReturnType<typeof dataApi.getStatistics>> }> =>
          r.status === 'fulfilled'
        )
        .map((r) => r.value)
    },
    enabled: selectedYears.length > 0,
  })

  // Generate comparison charts mutation
  const generateChartsMutation = useMutation({
    mutationFn: () => chartsApi.generateComparison(selectedYears),
  })

  const handleYearToggle = (year: string) => {
    setSelectedYears((prev) =>
      prev.includes(year)
        ? prev.filter((y) => y !== year)
        : [...prev, year].sort()
    )
  }

  // Prepare trend data
  const trendData = statsQueries.data?.map((item) => ({
    year: item.year,
    overall: item.stats?.weighted_averages?.['Overall Average'],
    grammar: item.stats?.weighted_averages?.['Grammar Average'],
    middle: item.stats?.weighted_averages?.['Middle Average'],
    high: item.stats?.weighted_averages?.['High Average'],
    responses: item.stats?.total_responses,
  })) ?? []

  // Prepare per-question data
  const questionData = QUESTIONS_SHORT.map(q => {
    const data: Record<string, unknown> = { question: q.label }
    statsQueries.data?.forEach(item => {
      // Find question totals that match this question
      const questionTotals = item.stats?.question_totals as Array<{
        question: string
        level: string
        weighted_avg: number | null
      }> | undefined

      if (questionTotals) {
        const matching = questionTotals.filter(qt =>
          qt.question?.includes(q.label.toLowerCase().split(' ')[0]) ||
          qt.question?.startsWith(`Q${q.key.slice(1)}`)
        )
        // Average across levels
        const validAvgs = matching
          .filter(m => m.weighted_avg !== null)
          .map(m => m.weighted_avg as number)
        if (validAvgs.length > 0) {
          data[item.year] = validAvgs.reduce((a, b) => a + b, 0) / validAvgs.length
        }
      }
    })
    return data
  })

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Compare Years</h2>
        {selectedYears.length >= 2 && (
          <Button
            onClick={() => generateChartsMutation.mutate()}
            disabled={generateChartsMutation.isPending}
            variant="outline"
          >
            {generateChartsMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <BarChart3 className="h-4 w-4 mr-2" />
                Generate Charts
              </>
            )}
          </Button>
        )}
      </div>

      {generateChartsMutation.isSuccess && (
        <Card className="bg-green-50 border-green-200">
          <CardContent className="py-4">
            <div className="text-green-800">
              Generated {generateChartsMutation.data.charts_generated} comparison charts.
              View them in the Visualizations tab under Artifacts.
            </div>
          </CardContent>
        </Card>
      )}

      {/* Year Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Select Years to Compare</CardTitle>
          <CardDescription>Choose multiple years to see trends</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            {yearsData?.years.map((year) => (
              <div key={year} className="flex items-center space-x-2">
                <Checkbox
                  id={year}
                  checked={selectedYears.includes(year)}
                  onCheckedChange={() => handleYearToggle(year)}
                />
                <label
                  htmlFor={year}
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  {year}
                </label>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {selectedYears.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            Select years above to compare survey results across time.
          </CardContent>
        </Card>
      ) : (
        <Tabs defaultValue="overview">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="questions">By Question</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Response Count Comparison */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Response Counts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-4">
                  {trendData.map((item, idx) => (
                    <div
                      key={item.year}
                      className="flex-1 p-4 border rounded-lg text-center"
                    >
                      <div
                        className="text-sm font-medium"
                        style={{ color: CHART_COLORS[idx % CHART_COLORS.length] }}
                      >
                        {item.year}
                      </div>
                      <div className="text-3xl font-bold mt-1">
                        {item.responses ?? '-'}
                      </div>
                      <div className="text-xs text-muted-foreground">responses</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Satisfaction Trends */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Satisfaction Trends</CardTitle>
                <CardDescription>
                  Average satisfaction scores over time (1-4 scale)
                </CardDescription>
              </CardHeader>
              <CardContent>
                {trendData.length > 0 ? (
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={trendData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <XAxis dataKey="year" />
                        <YAxis domain={[1, 4]} />
                        <Tooltip
                          formatter={(value: number) => value?.toFixed(2) ?? '-'}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="overall"
                          name="Overall"
                          stroke="#8884d8"
                          strokeWidth={2}
                          dot={{ r: 6 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="grammar"
                          name="Grammar"
                          stroke="#82ca9d"
                          strokeWidth={2}
                          dot={{ r: 6 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="middle"
                          name="Middle"
                          stroke="#ffc658"
                          strokeWidth={2}
                          dot={{ r: 6 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="high"
                          name="High"
                          stroke="#ff7300"
                          strokeWidth={2}
                          dot={{ r: 6 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    Loading statistics...
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Year-by-Year Comparison Table */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Detailed Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-4">Metric</th>
                        {trendData.map((item) => (
                          <th key={item.year} className="text-center py-2 px-4">
                            {item.year}
                          </th>
                        ))}
                        {trendData.length >= 2 && (
                          <th className="text-center py-2 px-4">Change</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b">
                        <td className="py-2 px-4 font-medium">Total Responses</td>
                        {trendData.map((item) => (
                          <td key={item.year} className="text-center py-2 px-4">
                            {item.responses ?? '-'}
                          </td>
                        ))}
                        {trendData.length >= 2 && (
                          <td className="text-center py-2 px-4">
                            {(() => {
                              const first = trendData[0]?.responses
                              const last = trendData[trendData.length - 1]?.responses
                              if (first && last) {
                                const change = last - first
                                return (
                                  <span className={change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : ''}>
                                    {change > 0 ? '+' : ''}{change}
                                  </span>
                                )
                              }
                              return '-'
                            })()}
                          </td>
                        )}
                      </tr>
                      {[
                        { key: 'overall', label: 'Overall Average' },
                        { key: 'grammar', label: 'Grammar Average' },
                        { key: 'middle', label: 'Middle Average' },
                        { key: 'high', label: 'High Average' },
                      ].map(({ key, label }) => (
                        <tr key={key} className="border-b">
                          <td className="py-2 px-4 font-medium">{label}</td>
                          {trendData.map((item) => (
                            <td key={item.year} className="text-center py-2 px-4">
                              {(item[key as keyof typeof item] as number)?.toFixed(2) ?? '-'}
                            </td>
                          ))}
                          {trendData.length >= 2 && (
                            <td className="text-center py-2 px-4">
                              {(() => {
                                const first = trendData[0]?.[key as keyof typeof trendData[0]] as number | undefined
                                const last = trendData[trendData.length - 1]?.[key as keyof typeof trendData[0]] as number | undefined
                                if (first && last) {
                                  const change = last - first
                                  return (
                                    <span className={change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : ''}>
                                      {change > 0 ? '+' : ''}{change.toFixed(2)}
                                    </span>
                                  )
                                }
                                return '-'
                              })()}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="questions" className="space-y-4">
            {/* Per-Question Comparison */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Year-over-Year by Question</CardTitle>
                <CardDescription>
                  Compare satisfaction scores for each question across years
                </CardDescription>
              </CardHeader>
              <CardContent>
                {questionData.length > 0 && selectedYears.length > 0 ? (
                  <div className="h-[500px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={questionData}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                      >
                        <XAxis type="number" domain={[0, 4]} />
                        <YAxis type="category" dataKey="question" width={140} />
                        <Tooltip
                          formatter={(value: number) => value?.toFixed(2) ?? '-'}
                        />
                        <Legend />
                        {selectedYears.map((year, idx) => (
                          <Bar
                            key={year}
                            dataKey={year}
                            name={year}
                            fill={CHART_COLORS[idx % CHART_COLORS.length]}
                          />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    Loading question data...
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Question Details Table */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Question Scores Detail</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-4">Question</th>
                        {selectedYears.map((year) => (
                          <th key={year} className="text-center py-2 px-4">
                            {year}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {questionData.map((row, idx) => (
                        <tr key={idx} className="border-b">
                          <td className="py-2 px-4 font-medium">{row.question as string}</td>
                          {selectedYears.map((year) => (
                            <td key={year} className="text-center py-2 px-4">
                              {(row[year] as number)?.toFixed(2) ?? '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}
