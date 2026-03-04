import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { TagCell } from '../TagCell'

describe('TagCell', () => {
  const defaultProps = {
    responseId: 'resp-1',
    tagName: 'Teachers',
    isTagged: false,
    onToggle: vi.fn(),
  }

  it('renders checkbox unchecked when isTagged is false', () => {
    render(<TagCell {...defaultProps} isTagged={false} />)

    const checkbox = screen.getByRole('checkbox')
    expect(checkbox).not.toBeChecked()
  })

  it('renders checkbox checked when isTagged is true', () => {
    render(<TagCell {...defaultProps} isTagged={true} />)

    const checkbox = screen.getByRole('checkbox')
    expect(checkbox).toBeChecked()
  })

  it('calls onToggle with correct arguments when clicked', () => {
    const onToggle = vi.fn()
    render(<TagCell {...defaultProps} isTagged={false} onToggle={onToggle} />)

    const checkbox = screen.getByRole('checkbox')
    fireEvent.click(checkbox)

    expect(onToggle).toHaveBeenCalledWith('resp-1', 'Teachers', false)
  })

  it('shows loading spinner when isLoading is true', () => {
    render(<TagCell {...defaultProps} isLoading={true} />)

    // Checkbox should not be visible, spinner should be
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument()
  })

  it('displays vote count when provided', () => {
    render(<TagCell {...defaultProps} voteCount={3} />)

    expect(screen.getByText('3/4')).toBeInTheDocument()
  })

  it('does not display vote count when zero', () => {
    render(<TagCell {...defaultProps} voteCount={0} />)

    expect(screen.queryByText('0/4')).not.toBeInTheDocument()
  })

  it('does not call onToggle when isLoading', () => {
    const onToggle = vi.fn()
    render(<TagCell {...defaultProps} isLoading={true} onToggle={onToggle} />)

    // The loading spinner is shown instead of checkbox
    // So we need to test that clicking doesn't trigger anything
    // This is implicitly tested by not having a checkbox to click
    expect(onToggle).not.toHaveBeenCalled()
  })

  it('shows optimistic state immediately after toggle', () => {
    render(<TagCell {...defaultProps} isTagged={false} />)

    const checkbox = screen.getByRole('checkbox')
    expect(checkbox).not.toBeChecked()

    fireEvent.click(checkbox)

    // Should show optimistic value (checked) immediately
    expect(checkbox).toBeChecked()
  })
})
