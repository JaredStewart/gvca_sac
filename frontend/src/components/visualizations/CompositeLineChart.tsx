import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { CsvColumn } from '@/lib/export-csv'

export interface CompositeDataPoint {
  year: string
  extremelySatisfied: number
  satisfied: number
  somewhatSatisfied: number
  dissatisfied: number
}

interface CompositeLineChartProps {
  data: CompositeDataPoint[]
}

const TIER_COLORS = {
  extremelySatisfied: '#4CAF50',
  satisfied: '#2196F3',
  somewhatSatisfied: '#FFC107',
  dissatisfied: '#B71C1C',
}

const TIER_LABELS: Record<string, string> = {
  extremelySatisfied: 'Extremely Satisfied',
  satisfied: 'Satisfied',
  somewhatSatisfied: 'Somewhat Satisfied',
  dissatisfied: 'Dissatisfied',
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function DataPointLabel({ x, y, value, stroke }: any) {
  if (value == null || value === 0) return null
  return (
    <text x={x} y={y} dy={-10} fill={stroke} fontSize={11} fontWeight="bold" textAnchor="middle">
      {Number(value).toFixed(1)}%
    </text>
  )
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload) return null
  return (
    <div className="rounded-md border bg-background p-2 shadow-sm text-xs">
      <p className="font-medium mb-1">{label}</p>
      {payload.map((entry: { value: number; dataKey: string; color: string }, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5 rounded-sm"
            style={{ backgroundColor: entry.color }}
          />
          <span>{TIER_LABELS[entry.dataKey]}: {Number(entry.value).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export default function CompositeLineChart({ data }: CompositeLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
        <XAxis dataKey="year" tick={{ fontSize: 12 }} />
        <YAxis
          domain={[0, 60]}
          tick={{ fontSize: 10 }}
          tickFormatter={(v: number) => `${v}%`}
          width={40}
        />
        <Tooltip content={CustomTooltip} />
        <Legend
          formatter={(value: string) => TIER_LABELS[value] ?? value}
          wrapperStyle={{ fontSize: 12 }}
        />
        {(Object.keys(TIER_COLORS) as Array<keyof typeof TIER_COLORS>).map((key) => (
          <Line
            key={key}
            type="linear"
            dataKey={key}
            stroke={TIER_COLORS[key]}
            strokeWidth={2.5}
            dot={{ r: 5, fill: TIER_COLORS[key] }}
            label={<DataPointLabel stroke={TIER_COLORS[key]} />}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

export function getCompositeCsvData(data: CompositeDataPoint[]): Record<string, unknown>[] {
  return data.map((d) => ({
    Year: d.year,
    'Extremely Satisfied (%)': d.extremelySatisfied,
    'Satisfied (%)': d.satisfied,
    'Somewhat Satisfied (%)': d.somewhatSatisfied,
    'Dissatisfied (%)': d.dissatisfied,
  }))
}

export const compositeCsvColumns: CsvColumn[] = [
  { header: 'Year', accessor: (r) => r['Year'] as string },
  { header: 'Extremely Satisfied (%)', accessor: (r) => Number(r['Extremely Satisfied (%)']).toFixed(1) },
  { header: 'Satisfied (%)', accessor: (r) => Number(r['Satisfied (%)']).toFixed(1) },
  { header: 'Somewhat Satisfied (%)', accessor: (r) => Number(r['Somewhat Satisfied (%)']).toFixed(1) },
  { header: 'Dissatisfied (%)', accessor: (r) => Number(r['Dissatisfied (%)']).toFixed(1) },
]
