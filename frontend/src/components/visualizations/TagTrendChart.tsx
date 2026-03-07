import { useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { CsvColumn } from '@/lib/export-csv'

export interface TagTrendDataPoint {
  year: string
  goodChoiceTotal: number
  betterServeTotal: number
  [key: string]: string | number // tag-specific keys like "goodChoice_Curriculum", "betterServe_Curriculum"
}

interface TagTrendChartProps {
  data: TagTrendDataPoint[]
  goodChoiceTags: string[]
  betterServeTags: string[]
}

// Same colors as Tag Frequency chart
const GOOD_CHOICE_COLOR = '#4CAF50'
const BETTER_SERVE_COLOR = '#800000'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload) return null
  const items = payload.filter((p: { dataKey: string; value: number }) => p.dataKey !== 'reset' && p.value !== 0)
  if (items.length === 0) return null
  return (
    <div className="rounded-md border bg-background p-2 shadow-sm text-xs">
      <p className="font-medium mb-1">{label}</p>
      {items.map((entry: { value: number; dataKey: string; color: string; name: string }, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5 rounded-sm"
            style={{ backgroundColor: entry.color }}
          />
          <span>{entry.name}: {Math.abs(entry.value)}</span>
        </div>
      ))}
    </div>
  )
}

export default function TagTrendChart({ data, goodChoiceTags, betterServeTags }: TagTrendChartProps) {
  // All unique tags for dropdown — default to first tag
  const allTags = Array.from(new Set([...goodChoiceTags, ...betterServeTags])).sort()
  const [selectedTag, setSelectedTag] = useState<string>(allTags[0] ?? '')

  const hasGc = goodChoiceTags.includes(selectedTag)
  const hasBs = betterServeTags.includes(selectedTag)

  // Build diverging bar data (same pattern as TagFrequencyChart)
  const chartData = data.map((point) => {
    const gcVal = hasGc ? ((point[`goodChoice_${selectedTag}`] as number) || 0) : 0
    const bsVal = hasBs ? Math.abs((point[`betterServe_${selectedTag}`] as number) || 0) : 0
    return {
      year: point.year,
      goodChoice: gcVal,
      reset: -gcVal,
      betterServe: -bsVal,
    }
  })

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-muted-foreground">Tag:</span>
        <Select value={selectedTag} onValueChange={setSelectedTag}>
          <SelectTrigger className="w-[220px] h-8 text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {allTags.map((tag) => (
              <SelectItem key={tag} value={tag}>{tag}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <ResponsiveContainer width="100%" height={420}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <XAxis dataKey="year" tick={{ fontSize: 12 }} />
          <YAxis
            tick={{ fontSize: 10 }}
            allowDecimals={false}
            tickFormatter={(v: number) => `${Math.abs(v)}`}
            width={40}
          />
          <Tooltip content={CustomTooltip} />
          <Legend
            payload={[
              { value: 'Good Choice (Q8)', type: 'square', color: GOOD_CHOICE_COLOR },
              { value: 'Better Serve (Q9)', type: 'square', color: BETTER_SERVE_COLOR },
            ]}
            wrapperStyle={{ fontSize: 12 }}
          />
          <ReferenceLine y={0} stroke="#666" strokeWidth={1.5} />
          <Bar dataKey="goodChoice" name="Good Choice" stackId="a" fill={GOOD_CHOICE_COLOR} isAnimationActive={false} />
          <Bar dataKey="reset" stackId="a" fill="transparent" isAnimationActive={false} legendType="none" />
          <Bar dataKey="betterServe" name="Better Serve" stackId="a" fill={BETTER_SERVE_COLOR} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function getTagTrendCsvData(
  data: TagTrendDataPoint[],
  goodChoiceTags: string[],
  betterServeTags: string[]
): Record<string, unknown>[] {
  return data.map((point) => {
    const row: Record<string, unknown> = { Year: point.year }
    for (const tag of goodChoiceTags) {
      row[`${tag} - Good Choice`] = (point[`goodChoice_${tag}`] as number) || 0
    }
    for (const tag of betterServeTags) {
      row[`${tag} - Better Serve`] = Math.abs((point[`betterServe_${tag}`] as number) || 0)
    }
    row['Good Choice Total'] = point.goodChoiceTotal
    row['Better Serve Total'] = point.betterServeTotal
    return row
  })
}

export function getTagTrendCsvColumns(
  goodChoiceTags: string[],
  betterServeTags: string[]
): CsvColumn[] {
  const cols: CsvColumn[] = [
    { header: 'Year', accessor: (r) => r['Year'] as string },
  ]
  for (const tag of goodChoiceTags) {
    cols.push({ header: `${tag} - Good Choice`, accessor: (r) => r[`${tag} - Good Choice`] as number })
  }
  for (const tag of betterServeTags) {
    cols.push({ header: `${tag} - Better Serve`, accessor: (r) => r[`${tag} - Better Serve`] as number })
  }
  cols.push({ header: 'Good Choice Total', accessor: (r) => r['Good Choice Total'] as number })
  cols.push({ header: 'Better Serve Total', accessor: (r) => r['Better Serve Total'] as number })
  return cols
}
