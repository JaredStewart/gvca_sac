import { useState, useCallback } from 'react'

interface Toast {
  id: string
  title?: string
  description?: string
  variant?: 'default' | 'destructive'
}

interface ToastState {
  toasts: Toast[]
}

// Simple toast implementation using console and state
export function useToast() {
  const [state, setState] = useState<ToastState>({ toasts: [] })

  const toast = useCallback(
    ({ title, description, variant }: Omit<Toast, 'id'>) => {
      const id = Math.random().toString(36).substring(2, 9)

      // Log to console for visibility
      if (variant === 'destructive') {
        console.error(`[Toast] ${title}: ${description}`)
      } else {
        console.log(`[Toast] ${title}: ${description}`)
      }

      // For now, also use alert for important messages
      if (variant === 'destructive') {
        alert(`Error: ${title}\n${description}`)
      }

      setState((prev) => ({
        toasts: [...prev.toasts, { id, title, description, variant }],
      }))

      // Auto-dismiss after 5 seconds
      setTimeout(() => {
        setState((prev) => ({
          toasts: prev.toasts.filter((t) => t.id !== id),
        }))
      }, 5000)

      return { id, dismiss: () => {} }
    },
    []
  )

  const dismiss = useCallback((toastId?: string) => {
    setState((prev) => ({
      toasts: toastId
        ? prev.toasts.filter((t) => t.id !== toastId)
        : [],
    }))
  }, [])

  return {
    toast,
    dismiss,
    toasts: state.toasts,
  }
}

export type { Toast }
