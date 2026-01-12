import { Question } from '../types';
import { questions as questionsData } from '../questions';

// Re-export the data for compatibility or simpler imports
export const questions: Question[] = questionsData as Question[];

/**
 * @deprecated Use questions.ts directly instead
 */
export function parseQuestions(_markdown?: string): Question[] {
  return questions;
}

export const RAW_DATA = "";