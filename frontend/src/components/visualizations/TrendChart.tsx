import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { ChartResult } from '@/api/client'

interface TrendChartProps {
  data: ChartResult['data']
  title?: string
  height?: number
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F']

export default function TrendChart({ data, title, height = 400 }: TrendChartProps) {
  // Transform data for the line chart
  // Group by year, with questions as separate lines
  const chartData = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) return []

    // Check if this is year-based trend data
    const isYearBased = 'year' in data[0]
    if (!isYearBased) return []

    // Group by year
    const byYear: Record<string, Record<string, number>> = {}
    const questions = new Set<string>()

    for (const item of data) {
      if (!('year' in item)) continue
      const year = item.year
      const question = item.question || 'Average'
      questions.add(question)

      if (!byYear[year]) {
        byYear[year] = {}
      }
      byYear[year][question] = item.average
    }

    // Convert to array format for recharts
    return Object.entries(byYear)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([year, values]) => ({
        year,
        ...values,
      }))
  }, [data])

  const questions = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) return []
    const qs = new Set<string>()
    for (const item of data) {
      if ('question' in item && item.question) {
        qs.add(item.question)
      }
    }
    return Array.from(qs)
  }, [data])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No trend data available
      </div>
    )
  }

  return (
    <div>
      {title && <h4 className="text-sm font-medium mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" />
          <YAxis domain={[0, 4]} tickCount={5} />
          <Tooltip
            formatter={(value: number) => value.toFixed(2)}
            labelFormatter={(label) => `Year: ${label}`}
          />
          <Legend />
          {questions.length === 0 ? (
            <Line
              type="monotone"
              dataKey="Average"
              stroke={COLORS[0]}
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          ) : (
            questions.map((question, index) => (
              <Line
                key={question}
                type="monotone"
                dataKey={question}
                name={question.length > 30 ? `${question.substring(0, 30)}...` : question}
                stroke={COLORS[index % COLORS.length]}
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            ))
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
