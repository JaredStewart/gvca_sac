import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts'

interface TagFrequencyData {
  tag: string
  goodChoiceCount: number
  betterServeCount: number
  totalCount: number
}

interface TagFrequencyChartProps {
  data: TagFrequencyData[]
}

interface ChartDatum {
  tag: string
  goodChoice: number
  reset: number
  betterServe: number
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload) return null
  return (
    <div className="rounded-md border bg-background p-2 shadow-sm text-xs">
      <p className="font-medium mb-1">{label}</p>
      {payload.filter((entry: { dataKey: string }) => entry.dataKey !== 'reset').map((entry: { value: number; dataKey: string; color: string }, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5 rounded-sm"
            style={{ backgroundColor: entry.color }}
          />
          <span>
            {entry.dataKey === 'goodChoice' ? 'Good Choice' : 'Better Serve'}:{' '}
            {Math.abs(entry.value)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function TagFrequencyChart({ data }: TagFrequencyChartProps) {
  const chartData: ChartDatum[] = data.map((d) => ({
    tag: d.tag,
    goodChoice: d.goodChoiceCount,
    reset: -d.goodChoiceCount,
    betterServe: -d.betterServeCount,
  }))

  return (
    <ResponsiveContainer width="100%" height={420}>
      <BarChart
        data={chartData}
        margin={{ top: 5, right: 20, left: 5, bottom: 80 }}
      >
        <XAxis
          dataKey="tag"
          tick={{ fontSize: 11 }}
          angle={-45}
          textAnchor="end"
          interval={0}
          height={80}
        />
        <YAxis
          tick={{ fontSize: 11 }}
          allowDecimals={false}
          tickFormatter={(v) => `${Math.abs(v)}`}
        />
        <Tooltip content={CustomTooltip} />
        <Legend
          payload={[
            { value: 'Good Choice (Q8)', type: 'square', color: '#4CAF50' },
            { value: 'Better Serve (Q9)', type: 'square', color: '#800000' },
          ]}
          wrapperStyle={{ fontSize: 12 }}
        />
        <ReferenceLine y={0} stroke="#666" strokeWidth={1} />
        <Bar dataKey="goodChoice" stackId="a" fill="#4CAF50" isAnimationActive={false} />
        <Bar dataKey="reset" stackId="a" fill="transparent" isAnimationActive={false} />
        <Bar dataKey="betterServe" stackId="a" fill="#800000" isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}
