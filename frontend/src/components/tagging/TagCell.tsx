import { useState, useEffect } from 'react'
import { Checkbox } from '@/components/ui/checkbox'
import { RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { DEFAULT_N_SAMPLES } from '@/constants'

interface TagCellProps {
  responseId: string
  tagName: string
  isTagged: boolean
  voteCount?: number
  isLoading?: boolean
  onToggle: (responseId: string, tagName: string, currentValue: boolean) => void
}

/**
 * A single tag cell in the tagging table with checkbox toggle.
 * Shows vote count from LLM generations (e.g., "3/4").
 */
export function TagCell({
  responseId,
  tagName,
  isTagged,
  voteCount,
  isLoading = false,
  onToggle,
}: TagCellProps) {
  const [optimisticValue, setOptimisticValue] = useState<boolean | null>(null)

  // Use optimistic value if set, otherwise use actual value
  const displayValue = optimisticValue !== null ? optimisticValue : isTagged

  const handleToggle = () => {
    if (isLoading) return

    // Set optimistic value immediately
    setOptimisticValue(!displayValue)

    // Call the toggle handler
    onToggle(responseId, tagName, displayValue)
  }

  // Reset optimistic value when actual value matches (server confirmed the change)
  useEffect(() => {
    if (optimisticValue !== null && optimisticValue === isTagged) {
      setOptimisticValue(null)
    }
  }, [isTagged, optimisticValue])

  return (
    <div className="flex flex-col items-center gap-1">
      {isLoading ? (
        <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
      ) : (
        <Checkbox
          checked={displayValue}
          onCheckedChange={handleToggle}
          className={cn(
            'h-5 w-5',
            optimisticValue !== null && 'opacity-60'
          )}
          disabled={isLoading}
        />
      )}
      {voteCount !== undefined && voteCount > 0 && (
        <span className="text-[10px] text-muted-foreground">
          {voteCount}/{DEFAULT_N_SAMPLES}
        </span>
      )}
    </div>
  )
}
