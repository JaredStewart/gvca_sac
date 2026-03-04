/**
 * Application-wide constants.
 */

// Chart colors (consistent across all visualizations)
export const CHART_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F']

// Stacked bar chart colors: top response → bottom response
export const STACKED_BAR_COLORS = {
  top: '#4CAF50',     // Green - best response
  second: '#2196F3',  // Blue
  third: '#FFC107',   // Yellow
  bottom: '#B71C1C',  // Maroon/Red - worst response
} as const

// Survey questions (Q1-Q7 scaled, Q8-Q9 free response)
// Presentation numbering: Q3-Q9 maps to indices 0-6
export const SURVEY_QUESTIONS = [
  { shortTitle: 'Education', presentationNumber: 3, fullText: 'How satisfied are you with the education that Golden View Classical Academy provided this year?' },
  { shortTitle: 'Intellectual Growth', presentationNumber: 4, fullText: 'Given your children\'s education level at the beginning of of the year, how satisfied are you with their intellectual growth this year?' },
  { shortTitle: 'School Culture', presentationNumber: 5, fullText: 'GVCA emphasizes 7 core virtues: Courage, Moderation, Justice, Responsibility, Prudence, Friendship, and Wonder. How well is the school culture reflected by these virtues?' },
  { shortTitle: 'Moral Character', presentationNumber: 6, fullText: 'How satisfied are you with your children\'s growth in moral character and civic virtue?' },
  { shortTitle: 'Teacher Communication', presentationNumber: 7, fullText: 'How effective is the communication between your family and your children\'s teachers?' },
  { shortTitle: 'Leadership Communication', presentationNumber: 8, fullText: 'How effective is the communication between your family and the school leadership?' },
  { shortTitle: 'Welcoming', presentationNumber: 9, fullText: 'How welcoming is the school community?' },
] as const

// Question-to-scale mapping (indices 0-6 map to Q1-Q7 in backend)
export type ScaleType = 'satisfaction' | 'reflection' | 'effectiveness' | 'welcoming'

export const QUESTION_SCALES: Record<number, { type: ScaleType; labels: string[] }> = {
  0: { type: 'satisfaction', labels: ['Extremely Satisfied', 'Satisfied', 'Somewhat Satisfied', 'Not Satisfied'] },
  1: { type: 'satisfaction', labels: ['Extremely Satisfied', 'Satisfied', 'Somewhat Satisfied', 'Not Satisfied'] },
  2: { type: 'reflection', labels: ['Strongly Reflected', 'Reflected', 'Somewhat Reflected', 'Not Reflected'] },
  3: { type: 'satisfaction', labels: ['Extremely Satisfied', 'Satisfied', 'Somewhat Satisfied', 'Not Satisfied'] },
  4: { type: 'effectiveness', labels: ['Extremely Effective', 'Effective', 'Somewhat Effective', 'Not Effective'] },
  5: { type: 'effectiveness', labels: ['Extremely Effective', 'Effective', 'Somewhat Effective', 'Not Effective'] },
  6: { type: 'welcoming', labels: ['Extremely Welcoming', 'Welcoming', 'Somewhat Welcoming', 'Not Welcoming'] },
}

// School levels
export const SCHOOL_LEVELS = ['Grammar', 'Middle', 'High'] as const

// Tagging constants (should match backend)
export const DEFAULT_N_SAMPLES = 4
export const STABILITY_THRESHOLD = 0.75

// Pagination constants
export const DEFAULT_PAGE_SIZE = 50
export const MAX_PAGE_SIZE = 200
