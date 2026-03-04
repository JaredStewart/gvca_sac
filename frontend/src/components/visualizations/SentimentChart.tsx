import { useMemo, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts'
import type { SentimentChartResult } from '@/api/client'

interface SentimentChartProps {
  data: SentimentChartResult['data']
  height?: number
  onBarClick?: (tag: string, sentiment: 'positive' | 'negative') => void
}

export default function SentimentChart({
  data,
  height = 400,
  onBarClick,
}: SentimentChartProps) {
  const [hoveredBar, setHoveredBar] = useState<string | null>(null)

  // Transform data for diverging bar chart
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []

    return data
      .map((item) => ({
        tag: item.tag,
        positive: item.positive_count,
        negative: -item.negative_count, // Negative values for left side
        net: item.net_sentiment,
      }))
      .sort((a, b) => b.net - a.net) // Sort by net sentiment
  }, [data])

  // Calculate max value for symmetric axis
  const maxValue = useMemo(() => {
    if (chartData.length === 0) return 10
    const max = Math.max(
      ...chartData.map((d) => Math.max(d.positive, Math.abs(d.negative)))
    )
    return Math.ceil(max * 1.1) // Add 10% padding
  }, [chartData])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No sentiment data available
      </div>
    )
  }

  const handleClick = (entry: { tag: string }, sentiment: 'positive' | 'negative') => {
    if (onBarClick) {
      onBarClick(entry.tag, sentiment)
    }
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          domain={[-maxValue, maxValue]}
          tickFormatter={(value) => Math.abs(value).toString()}
        />
        <YAxis
          type="category"
          dataKey="tag"
          width={110}
          tick={{ fontSize: 12 }}
        />
        <Tooltip
          formatter={(value: number, name: string) => [
            Math.abs(value),
            name === 'negative' ? 'Improvement mentions' : 'Praise mentions',
          ]}
          labelFormatter={(label) => `Tag: ${label}`}
        />
        <ReferenceLine x={0} stroke="#666" />

        {/* Negative (improvement) bars - left side */}
        <Bar
          dataKey="negative"
          name="negative"
          stackId="stack"
          onClick={(entry) => handleClick(entry as { tag: string }, 'negative')}
          onMouseEnter={(data) => setHoveredBar(`${data.tag}-negative`)}
          onMouseLeave={() => setHoveredBar(null)}
          cursor={onBarClick ? 'pointer' : 'default'}
        >
          {chartData.map((entry) => (
            <Cell
              key={`negative-${entry.tag}`}
              fill={hoveredBar === `${entry.tag}-negative` ? '#dc2626' : '#ef4444'}
            />
          ))}
        </Bar>

        {/* Positive (praise) bars - right side */}
        <Bar
          dataKey="positive"
          name="positive"
          stackId="stack"
          onClick={(entry) => handleClick(entry as { tag: string }, 'positive')}
          onMouseEnter={(data) => setHoveredBar(`${data.tag}-positive`)}
          onMouseLeave={() => setHoveredBar(null)}
          cursor={onBarClick ? 'pointer' : 'default'}
        >
          {chartData.map((entry) => (
            <Cell
              key={`positive-${entry.tag}`}
              fill={hoveredBar === `${entry.tag}-positive` ? '#16a34a' : '#22c55e'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
