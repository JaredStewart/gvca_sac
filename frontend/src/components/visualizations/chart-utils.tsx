import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts'
import { STACKED_BAR_COLORS } from '@/constants'

export const COLORS = [
  STACKED_BAR_COLORS.top,
  STACKED_BAR_COLORS.second,
  STACKED_BAR_COLORS.third,
  STACKED_BAR_COLORS.bottom,
]

export interface StackedBarDatum {
  name: string
  seg0: number
  seg1: number
  seg2: number
  seg3: number
  weightedAverage: number
}

export function roundDistribution(dist: number[]): number[] {
  const rounded = dist.map((v) => Math.round(v * 10) / 10)
  const sum = rounded.reduce((a, b) => a + b, 0)
  const diff = Math.round((sum - 100) * 10) / 10
  if (diff !== 0 && rounded.length > 0) {
    let maxIdx = 0
    for (let i = 1; i < rounded.length; i++) {
      if (rounded[i] > rounded[maxIdx]) maxIdx = i
    }
    rounded[maxIdx] = Math.round((rounded[maxIdx] - diff) * 10) / 10
  }
  return rounded
}

export function toStackedData(
  bars: Array<{ name: string; distribution: number[]; weightedAverage: number }>
): StackedBarDatum[] {
  return bars.map((b) => {
    const rd = roundDistribution(b.distribution.map((v) => v ?? 0))
    return {
      name: b.name,
      seg0: rd[0],
      seg1: rd[1],
      seg2: rd[2],
      seg3: rd[3],
      weightedAverage: b.weightedAverage,
    }
  })
}

function CustomXAxisTick({ x, y, payload, data }: {
  x?: number
  y?: number
  payload?: { value: string }
  data: StackedBarDatum[]
}) {
  const item = data.find((d) => d.name === payload?.value)
  const avg = item?.weightedAverage
  return (
    <g transform={`translate(${x},${y})`}>
      <text x={0} y={0} dy={12} textAnchor="middle" fill="#666" fontSize={12}>
        {payload?.value}
      </text>
      {avg !== undefined && avg > 0 && (
        <text x={0} y={0} dy={26} textAnchor="middle" fill="#333" fontSize={11} fontWeight="bold">
          ({avg.toFixed(2)})
        </text>
      )}
    </g>
  )
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label, scaleLabels }: any & { scaleLabels: string[] }) {
  if (!active || !payload) return null
  return (
    <div className="rounded-md border bg-background p-2 shadow-sm text-xs">
      <p className="font-medium mb-1">{label}</p>
      {[...payload].reverse().map((entry: { value: number; dataKey: string }, i: number) => {
        const segIndex = parseInt(entry.dataKey.replace('seg', ''))
        return (
          <div key={i} className="flex items-center gap-2">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm"
              style={{ backgroundColor: COLORS[segIndex] }}
            />
            <span>{scaleLabels[segIndex]}: {entry.value.toFixed(1)}%</span>
          </div>
        )
      })}
    </div>
  )
}

export function StackedBarSection({
  title,
  data,
  scaleLabels,
  height = 260,
}: {
  title?: string
  data: StackedBarDatum[]
  scaleLabels: string[]
  height?: number
}) {
  return (
    <div className="w-full">
      {title && (
        <h4 className="text-xs font-medium text-muted-foreground mb-1 text-center">{title}</h4>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 35 }}>
          <XAxis
            dataKey="name"
            tick={(props) => <CustomXAxisTick {...props} data={data} />}
            interval={0}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={[0, 100]}
            allowDataOverflow={true}
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => `${Math.round(v)}%`}
            width={35}
          />
          <Tooltip content={(props) => <CustomTooltip {...props} scaleLabels={scaleLabels} />} />
          <ReferenceLine y={0} stroke="#e5e7eb" />
          <Bar dataKey="seg3" stackId="stack" isAnimationActive={false}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[3]} />
            ))}
          </Bar>
          <Bar dataKey="seg2" stackId="stack" isAnimationActive={false}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[2]} />
            ))}
          </Bar>
          <Bar dataKey="seg1" stackId="stack" isAnimationActive={false}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[1]} />
            ))}
          </Bar>
          <Bar dataKey="seg0" stackId="stack" isAnimationActive={false}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[0]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function ScaleLegend({ scaleLabels }: { scaleLabels: string[] }) {
  return (
    <div className="flex flex-wrap gap-3 mb-2 text-xs">
      {scaleLabels.map((label, i) => (
        <div key={label} className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-3 rounded-sm"
            style={{ backgroundColor: COLORS[i] }}
          />
          <span>{label}</span>
        </div>
      ))}
    </div>
  )
}
