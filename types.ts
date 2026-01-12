export interface Question {
  id: string;
  category: string;
  text: string;
  options: string[];
  correctIndices: number[]; // Array of indices (0 for A, 1 for B...)
}

export interface QuestionStats {
  questionId: string;
  streak: number; // How many times correctly answered in a row
  attempts: number;
  lastAnswered: number; // Timestamp
}

export interface AppState {
  questions: Question[];
  stats: Record<string, QuestionStats>;
  currentQuestionId: string | null;
  isSessionActive: boolean;
}