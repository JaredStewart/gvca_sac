import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { taggingApi, dataApi } from '@/api/client'
import { useAppStore } from '@/stores/app'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const LEVEL_SEGMENTS = [
  { value: 'all', label: 'All Responses' },
  { value: 'Grammar', label: 'Grammar School' },
  { value: 'Middle', label: 'Middle School' },
  { value: 'High', label: 'High School' },
]

const COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F',
  '#FFBB28', '#FF8042', '#0088FE', '#A569BD', '#45B39D',
]

export default function Segments() {
  const { selectedYear } = useAppStore()
  const [segment1, setSegment1] = useState('all')
  const [segment2, setSegment2] = useState('Grammar')
  const [selectedDemoSegment, setSelectedDemoSegment] = useState('year_1')

  // Fetch available demographic segments
  const { data: segments } = useQuery({
    queryKey: ['segments', selectedYear],
    queryFn: () => selectedYear ? dataApi.getSegments(selectedYear) : null,
    enabled: !!selectedYear,
  })

  // Fetch segment statistics for comparison
  const { data: segmentStats } = useQuery({
    queryKey: ['segment-stats', selectedYear, selectedDemoSegment],
    queryFn: () => selectedYear ? dataApi.getSegmentStatistics(selectedYear, selectedDemoSegment) : null,
    enabled: !!selectedYear && !!selectedDemoSegment,
  })

  const { data: inverseStats } = useQuery({
    queryKey: ['segment-stats-inverse', selectedYear, selectedDemoSegment],
    queryFn: () => selectedYear ? dataApi.getSegmentStatistics(selectedYear, selectedDemoSegment, true) : null,
    enabled: !!selectedYear && !!selectedDemoSegment,
  })

  // Fetch tag distributions for level comparison
  const { data: dist1 } = useQuery({
    queryKey: ['tag-distribution', selectedYear, segment1],
    queryFn: () => selectedYear ? taggingApi.getDistribution(selectedYear, segment1 === 'all' ? undefined : segment1) : null,
    enabled: !!selectedYear,
  })

  const { data: dist2 } = useQuery({
    queryKey: ['tag-distribution', selectedYear, segment2],
    queryFn: () => selectedYear ? taggingApi.getDistribution(selectedYear, segment2 || undefined) : null,
    enabled: !!selectedYear && !!segment2,
  })

  if (!selectedYear) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <Card className="w-96">
          <CardHeader>
            <CardTitle>No Year Selected</CardTitle>
            <CardDescription>
              Select a survey year from the dropdown to analyze segments.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  // Prepare level comparison data
  const comparisonData = dist1?.distribution.map((item) => {
    const match = dist2?.distribution.find((d) => d.tag === item.tag)
    return {
      tag: item.tag,
      segment1: item.percentage,
      segment2: match?.percentage ?? 0,
    }
  }).sort((a, b) => b.segment1 - a.segment1) ?? []

  // Prepare demographic comparison data
  const demoComparisonData = segmentStats && inverseStats ? [
    {
      metric: 'Overall Average',
      segment: segmentStats.statistics.weighted_averages?.['Overall Average'] ?? 0,
      inverse: inverseStats.statistics.weighted_averages?.['Overall Average'] ?? 0,
    },
    {
      metric: 'Grammar',
      segment: segmentStats.statistics.weighted_averages?.['Grammar Average'] ?? 0,
      inverse: inverseStats.statistics.weighted_averages?.['Grammar Average'] ?? 0,
    },
    {
      metric: 'Middle',
      segment: segmentStats.statistics.weighted_averages?.['Middle Average'] ?? 0,
      inverse: inverseStats.statistics.weighted_averages?.['Middle Average'] ?? 0,
    },
    {
      metric: 'High',
      segment: segmentStats.statistics.weighted_averages?.['High Average'] ?? 0,
      inverse: inverseStats.statistics.weighted_averages?.['High Average'] ?? 0,
    },
  ] : []

  const currentSegmentConfig = segments?.segments.find(s => s.key === selectedDemoSegment)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Segment Analysis</h2>
      </div>

      <Tabs defaultValue="demographic">
        <TabsList>
          <TabsTrigger value="demographic">Demographic Segments</TabsTrigger>
          <TabsTrigger value="level">By School Level</TabsTrigger>
        </TabsList>

        <TabsContent value="demographic" className="space-y-4">
          {/* Demographic Segment Overview */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Demographic Segments</CardTitle>
              <CardDescription>
                Compare satisfaction across demographic groups
              </CardDescription>
            </CardHeader>
            <CardContent>
              {segments?.segments && segments.segments.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  {segments.segments.map((seg) => (
                    <div
                      key={seg.key}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedDemoSegment === seg.key ? 'border-primary bg-primary/5' : 'hover:bg-muted'
                      }`}
                      onClick={() => setSelectedDemoSegment(seg.key)}
                    >
                      <div className="text-sm font-medium">{seg.name}</div>
                      <div className="text-2xl font-bold">{seg.count}</div>
                      <div className="text-xs text-muted-foreground">
                        {seg.percentage}% of responses
                      </div>
                      <div className="mt-2 text-xs text-muted-foreground">
                        {seg.inverse_name}: {seg.inverse_count} ({seg.inverse_percentage}%)
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No segment data available. Load data from the Dashboard.
                </div>
              )}
            </CardContent>
          </Card>

          {/* Segment Comparison Chart */}
          {currentSegmentConfig && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">
                  {currentSegmentConfig.name} vs {currentSegmentConfig.inverse_name}
                </CardTitle>
                <CardDescription>
                  Average satisfaction scores (1-4 scale)
                </CardDescription>
              </CardHeader>
              <CardContent>
                {demoComparisonData.length > 0 ? (
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={demoComparisonData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <XAxis dataKey="metric" />
                        <YAxis domain={[0, 4]} />
                        <Tooltip
                          formatter={(value: number) => value?.toFixed(2) ?? '-'}
                        />
                        <Legend />
                        <Bar
                          dataKey="segment"
                          name={currentSegmentConfig.name}
                          fill="#8884d8"
                        />
                        <Bar
                          dataKey="inverse"
                          name={currentSegmentConfig.inverse_name}
                          fill="#82ca9d"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    Loading comparison data...
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Detailed Comparison Table */}
          {segmentStats && inverseStats && currentSegmentConfig && (
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
                        <th className="text-center py-2 px-4">
                          {currentSegmentConfig.name}
                          <div className="text-xs font-normal text-muted-foreground">
                            n={segmentStats.statistics.total_responses}
                          </div>
                        </th>
                        <th className="text-center py-2 px-4">
                          {currentSegmentConfig.inverse_name}
                          <div className="text-xs font-normal text-muted-foreground">
                            n={inverseStats.statistics.total_responses}
                          </div>
                        </th>
                        <th className="text-center py-2 px-4">Difference</th>
                      </tr>
                    </thead>
                    <tbody>
                      {['Overall Average', 'Grammar Average', 'Middle Average', 'High Average'].map((metric) => {
                        const val1 = segmentStats.statistics.weighted_averages?.[metric]
                        const val2 = inverseStats.statistics.weighted_averages?.[metric]
                        const diff = val1 && val2 ? val1 - val2 : null

                        return (
                          <tr key={metric} className="border-b">
                            <td className="py-2 px-4 font-medium">{metric.replace(' Average', '')}</td>
                            <td className="text-center py-2 px-4">
                              {val1?.toFixed(2) ?? '-'}
                            </td>
                            <td className="text-center py-2 px-4">
                              {val2?.toFixed(2) ?? '-'}
                            </td>
                            <td className={`text-center py-2 px-4 ${
                              diff && diff > 0 ? 'text-green-600' :
                              diff && diff < 0 ? 'text-red-600' : ''
                            }`}>
                              {diff !== null ? (diff > 0 ? '+' : '') + diff.toFixed(2) : '-'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="level" className="space-y-4">
          {/* Level Segment Selectors */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Compare by School Level</CardTitle>
              <CardDescription>Select two levels to compare tag distributions</CardDescription>
            </CardHeader>
            <CardContent className="flex gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Segment 1</label>
                <Select value={segment1} onValueChange={setSegment1}>
                  <SelectTrigger className="w-[200px]">
                    <SelectValue placeholder="Select segment" />
                  </SelectTrigger>
                  <SelectContent>
                    {LEVEL_SEGMENTS.map((seg) => (
                      <SelectItem key={seg.value} value={seg.value}>
                        {seg.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Segment 2</label>
                <Select value={segment2} onValueChange={setSegment2}>
                  <SelectTrigger className="w-[200px]">
                    <SelectValue placeholder="Select segment" />
                  </SelectTrigger>
                  <SelectContent>
                    {LEVEL_SEGMENTS.filter(s => s.value !== segment1).map((seg) => (
                      <SelectItem key={seg.value} value={seg.value}>
                        {seg.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Level Distribution Comparison Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Tag Distribution Comparison</CardTitle>
              <CardDescription>
                Percentage of responses with each tag
              </CardDescription>
            </CardHeader>
            <CardContent>
              {comparisonData.length > 0 ? (
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={comparisonData.slice(0, 10)}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                    >
                      <XAxis type="number" domain={[0, 100]} unit="%" />
                      <YAxis type="category" dataKey="tag" width={90} />
                      <Tooltip
                        formatter={(value: number) => `${value.toFixed(1)}%`}
                      />
                      <Legend />
                      <Bar
                        dataKey="segment1"
                        name={LEVEL_SEGMENTS.find(s => s.value === segment1)?.label || 'All'}
                        fill="#8884d8"
                      />
                      <Bar
                        dataKey="segment2"
                        name={LEVEL_SEGMENTS.find(s => s.value === segment2)?.label || segment2}
                        fill="#82ca9d"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No tagging data available. Run tagging from the Dashboard.
                </div>
              )}
            </CardContent>
          </Card>

          {/* Individual Level Stats */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">
                  {LEVEL_SEGMENTS.find(s => s.value === segment1)?.label || 'All Responses'}
                </CardTitle>
                <CardDescription>
                  {dist1?.unique_tags ?? 0} unique tags
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {dist1?.distribution.slice(0, 10).map((item, idx) => (
                    <Badge
                      key={item.tag}
                      style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                      className="text-white"
                    >
                      {item.tag}: {item.percentage.toFixed(1)}%
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">
                  {LEVEL_SEGMENTS.find(s => s.value === segment2)?.label || segment2}
                </CardTitle>
                <CardDescription>
                  {dist2?.unique_tags ?? 0} unique tags
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {dist2?.distribution.slice(0, 10).map((item, idx) => (
                    <Badge
                      key={item.tag}
                      style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                      className="text-white"
                    >
                      {item.tag}: {item.percentage.toFixed(1)}%
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
