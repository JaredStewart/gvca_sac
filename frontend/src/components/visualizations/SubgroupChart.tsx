import { toStackedData, StackedBarSection, ScaleLegend } from './chart-utils'
import type { CsvColumn } from '@/lib/export-csv'

export interface SubgroupBar {
  level: string
  distribution: number[]
  counts: number[]
  weightedAverage: number
}

export interface SubgroupChartProps {
  presentationNumber: number
  shortTitle: string
  scaleLabels: string[]
  subgroups: SubgroupBar[]
  selectedYear: string
  historicalAverages: Record<string, Record<string, number>>
}

export default function SubgroupChart({
  presentationNumber,
  shortTitle,
  scaleLabels,
  subgroups,
  selectedYear,
  historicalAverages,
}: SubgroupChartProps) {
  const data = toStackedData(
    subgroups.map((s) => ({ name: s.level, distribution: s.distribution, weightedAverage: s.weightedAverage }))
  )

  const histYears = Object.keys(historicalAverages).sort()
  const otherYears = histYears.filter((y) => y !== selectedYear)

  return (
    <div>
      <h3 className="text-sm font-semibold mb-2">
        Q{presentationNumber}: {shortTitle} - {selectedYear} Subgroup Responses
      </h3>
      <ScaleLegend scaleLabels={scaleLabels} />
      <StackedBarSection data={data} scaleLabels={scaleLabels} />
      {otherYears.length > 0 && (
        <div className="mt-1 text-xs text-muted-foreground">
          {subgroups.map((sg) => {
            const avgs = otherYears
              .map((y) => {
                const val = historicalAverages[y]?.[sg.level]
                return val != null && val > 0 ? `${y}: ${val.toFixed(2)}` : null
              })
              .filter(Boolean)
            if (avgs.length === 0) return null
            return (
              <span key={sg.level} className="mr-4">
                {sg.level} ({avgs.join(', ')})
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}

export function getSubgroupCsvData(
  subgroups: SubgroupBar[],
  scaleLabels: string[]
): Record<string, unknown>[] {
  return subgroups.map((sg) => {
    const row: Record<string, unknown> = { Level: sg.level }
    for (let i = 0; i < scaleLabels.length; i++) {
      row[`${scaleLabels[i]} (%)`] = sg.distribution[i] ?? 0
      row[`${scaleLabels[i]} (N)`] = sg.counts[i] ?? 0
    }
    row['Weighted Average'] = sg.weightedAverage
    return row
  })
}

export function getSubgroupCsvColumns(scaleLabels: string[]): CsvColumn[] {
  const cols: CsvColumn[] = [
    { header: 'Level', accessor: (r) => r['Level'] as string },
  ]
  for (const label of scaleLabels) {
    cols.push({ header: `${label} (%)`, accessor: (r) => Number(r[`${label} (%)`]).toFixed(1) })
    cols.push({ header: `${label} (N)`, accessor: (r) => r[`${label} (N)`] as number })
  }
  cols.push({ header: 'Weighted Average', accessor: (r) => Number(r['Weighted Average']).toFixed(2) })
  return cols
}
