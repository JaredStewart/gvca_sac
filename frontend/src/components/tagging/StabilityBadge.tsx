import { Badge } from '@/components/ui/badge'

interface StabilityBadgeProps {
  score: number | null
  showTooltip?: boolean
}

/**
 * Displays a color-coded stability score badge.
 * - Green (≥0.8): High stability - consistent tagging across LLM runs
 * - Yellow (≥0.6): Medium stability - some variation in tagging
 * - Red (<0.6): Low stability - needs human review
 */
export function StabilityBadge({ score, showTooltip = true }: StabilityBadgeProps) {
  if (score === null || score === undefined) {
    return (
      <Badge variant="outline" className="text-muted-foreground">
        N/A
      </Badge>
    )
  }

  const percentage = Math.round(score * 100)

  // Determine color based on threshold
  let colorClasses: string
  let label: string

  if (score >= 0.8) {
    colorClasses = 'bg-green-100 text-green-800 border-green-200 hover:bg-green-200'
    label = 'High'
  } else if (score >= 0.6) {
    colorClasses = 'bg-yellow-100 text-yellow-800 border-yellow-200 hover:bg-yellow-200'
    label = 'Medium'
  } else {
    colorClasses = 'bg-red-100 text-red-800 border-red-200 hover:bg-red-200'
    label = 'Low'
  }

  const badge = (
    <Badge
      variant="outline"
      className={`${colorClasses} font-medium`}
    >
      {percentage}%
    </Badge>
  )

  if (!showTooltip) {
    return badge
  }

  return (
    <div className="relative group inline-block">
      {badge}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10 pointer-events-none">
        {label} stability: {percentage}% avg IoU across 4 LLM runs
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
      </div>
    </div>
  )
}
