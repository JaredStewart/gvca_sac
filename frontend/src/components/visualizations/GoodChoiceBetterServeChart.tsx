import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  LabelList,
} from 'recharts'
import type { CsvColumn } from '@/lib/export-csv'

export interface FreeResponseCounts {
  total_good_choice: number
  total_better_serve: number
  only_good_choice: number
  only_better_serve: number
  both: number
  only_positive_pct: number
}

interface GoodChoiceBetterServeChartProps {
  data: FreeResponseCounts
}

const BAR_COLORS = {
  goodChoice: '#4CAF50',
  onlyGoodChoice: '#A5D6A7',
  betterServe: '#800020',
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload) return null
  const entry = payload[0]
  if (!entry) return null
  return (
    <div className="rounded-md border bg-background p-2 shadow-sm text-xs">
      <p className="font-medium mb-1">{label}</p>
      <p>{entry.value} responses</p>
    </div>
  )
}

export default function GoodChoiceBetterServeChart({ data }: GoodChoiceBetterServeChartProps) {
  // Waterfall layout: "ONLY Good Choice" hangs from the top of "Good Choice" bar
  const chartData = [
    {
      name: 'Good Choice\nResponses',
      base: 0,
      value: data.total_good_choice,
      color: BAR_COLORS.goodChoice,
      label: data.total_good_choice,
    },
    {
      name: 'ONLY Good Choice\nResponses Provided',
      base: data.both, // invisible base: starts at "both" level
      value: data.only_good_choice, // visible portion: the difference
      color: BAR_COLORS.onlyGoodChoice,
      label: data.only_good_choice,
    },
    {
      name: 'Better Serve\nResponses',
      base: 0,
      value: data.total_better_serve,
      color: BAR_COLORS.betterServe,
      label: data.total_better_serve,
    },
  ]

  const maxVal = data.total_good_choice * 1.15

  return (
    <ResponsiveContainer width="100%" height={380}>
      <BarChart data={chartData} margin={{ top: 40, right: 30, left: 10, bottom: 30 }}>
        <XAxis
          dataKey="name"
          tick={{ fontSize: 11 }}
          interval={0}
          tickLine={false}
        />
        <YAxis
          domain={[0, Math.ceil(maxVal / 50) * 50]}
          tick={{ fontSize: 10 }}
          label={{ value: 'Number of Open Responses', angle: -90, position: 'insideLeft', offset: 0, fontSize: 11 }}
          width={50}
        />
        <Tooltip content={CustomTooltip} />
        <ReferenceLine y={data.total_good_choice} stroke="#999" strokeDasharray="5 5" />
        <ReferenceLine y={data.total_better_serve} stroke="#999" strokeDasharray="5 5" />
        {/* Invisible base bar */}
        <Bar dataKey="base" stackId="waterfall" fill="transparent" isAnimationActive={false} maxBarSize={100} />
        {/* Visible value bar stacked on top of base */}
        <Bar dataKey="value" stackId="waterfall" isAnimationActive={false} maxBarSize={100}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.color} />
          ))}
          <LabelList
            dataKey="label"
            position="inside"
            fill="white"
            fontWeight="bold"
            fontSize={18}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function getGoodChoiceBetterServeCsvData(data: FreeResponseCounts): Record<string, unknown>[] {
  return [
    { Category: 'Good Choice Responses', Count: data.total_good_choice },
    { Category: 'ONLY Good Choice Responses Provided', Count: data.only_good_choice },
    { Category: 'Better Serve Responses', Count: data.total_better_serve },
    { Category: 'ONLY Better Serve Responses Provided', Count: data.only_better_serve },
    { Category: 'Both Good Choice & Better Serve', Count: data.both },
    { Category: 'ONLY Positive Response %', Count: data.only_positive_pct },
  ]
}

export const goodChoiceBetterServeCsvColumns: CsvColumn[] = [
  { header: 'Category', accessor: (r) => r['Category'] as string },
  { header: 'Count', accessor: (r) => r['Count'] as number },
]
