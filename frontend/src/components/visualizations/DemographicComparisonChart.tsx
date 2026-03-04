import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { ChartResult } from '@/api/client'

interface DemographicComparisonChartProps {
  data: ChartResult['data']
  segmentALabel: string
  segmentBLabel: string
  height?: number
}

export default function DemographicComparisonChart({
  data,
  segmentALabel,
  segmentBLabel,
  height = 400,
}: DemographicComparisonChartProps) {
  const chartData = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) return []

    return data
      .filter((item): item is Extract<typeof item, { question: string }> =>
        'question' in item && 'segment_a_avg' in item
      )
      .map((item) => ({
        question:
          item.question.length > 25
            ? `${item.question.substring(0, 25)}...`
            : item.question,
        fullQuestion: item.question,
        [segmentALabel]: item.segment_a_avg,
        [segmentBLabel]: item.segment_b_avg,
        difference: item.difference,
      }))
  }, [data, segmentALabel, segmentBLabel])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No comparison data available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 20, right: 30, left: 150, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 4]} tickCount={5} />
        <YAxis
          type="category"
          dataKey="question"
          width={140}
          tick={{ fontSize: 11 }}
        />
        <Tooltip
          formatter={(value: number) => value.toFixed(2)}
          labelFormatter={(_, payload) => {
            if (payload && payload[0]) {
              return payload[0].payload.fullQuestion
            }
            return ''
          }}
        />
        <Legend />
        <Bar dataKey={segmentALabel} fill="#8884d8" />
        <Bar dataKey={segmentBLabel} fill="#82ca9d" />
      </BarChart>
    </ResponsiveContainer>
  )
}
