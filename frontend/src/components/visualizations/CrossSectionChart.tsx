import { toStackedData, StackedBarSection, ScaleLegend } from './chart-utils'
import type { CsvColumn } from '@/lib/export-csv'

export interface CrossSectionBar {
  questionLabel: string
  distribution: number[]
  weightedAverage: number
}

export interface CrossSectionChartProps {
  title: string
  bars: CrossSectionBar[]
}

// Generic tier labels since questions span different scales (satisfaction, reflection, effectiveness, welcoming)
const CROSS_SECTION_SCALE_LABELS = [
  'Strongly Positive',
  'Positive',
  'Somewhat Positive',
  'Not Positive',
]

export default function CrossSectionChart({ title, bars }: CrossSectionChartProps) {
  const data = toStackedData(
    bars.map((b) => ({ name: b.questionLabel, distribution: b.distribution, weightedAverage: b.weightedAverage }))
  )

  return (
    <div>
      <h3 className="text-sm font-semibold mb-2">{title}</h3>
      <ScaleLegend scaleLabels={CROSS_SECTION_SCALE_LABELS} />
      <StackedBarSection data={data} scaleLabels={CROSS_SECTION_SCALE_LABELS} height={300} />
    </div>
  )
}

export function getCrossSectionCsvData(bars: CrossSectionBar[]): Record<string, unknown>[] {
  return bars.map((b) => ({
    Question: b.questionLabel,
    'Strongly Positive (%)': b.distribution[0] ?? 0,
    'Positive (%)': b.distribution[1] ?? 0,
    'Somewhat Positive (%)': b.distribution[2] ?? 0,
    'Not Positive (%)': b.distribution[3] ?? 0,
    'Weighted Average': b.weightedAverage,
  }))
}

export const crossSectionCsvColumns: CsvColumn[] = [
  { header: 'Question', accessor: (r) => r['Question'] as string },
  { header: 'Strongly Positive (%)', accessor: (r) => Number(r['Strongly Positive (%)']).toFixed(1) },
  { header: 'Positive (%)', accessor: (r) => Number(r['Positive (%)']).toFixed(1) },
  { header: 'Somewhat Positive (%)', accessor: (r) => Number(r['Somewhat Positive (%)']).toFixed(1) },
  { header: 'Not Positive (%)', accessor: (r) => Number(r['Not Positive (%)']).toFixed(1) },
  { header: 'Weighted Average', accessor: (r) => Number(r['Weighted Average']).toFixed(2) },
]
