import { Question, QuestionStats } from '../types';

/**
 * Weighted Random Selection for Spaced Repetition
 * 
 * Logic:
 * 1. Filter valid candidates.
 * 2. Calculate weight for each question based on streak (mastery).
 *    - New questions (no stats): High Weight (e.g., 5)
 *    - Wrong recently (streak 0): High Weight (e.g., 5)
 *    - Streak 1: Weight 3
 *    - Streak 2: Weight 2
 *    - Streak 3+: Weight 1 (Review rarely)
 * 3. Pick random based on weights.
 */
export function getNextQuestion(
  allQuestions: Question[], 
  stats: Record<string, QuestionStats>,
  lastQuestionId: string | null
): Question | null {
  
  if (allQuestions.length === 0) return null;

  const candidates = allQuestions.map(q => {
    const s = stats[q.id];
    let weight = 5; // Default for new questions

    if (s) {
      if (s.streak === 0) weight = 6; // Prioritize mistakes heavily
      else if (s.streak === 1) weight = 4;
      else if (s.streak === 2) weight = 2;
      else weight = 1; // Mastered items appear less
    }

    // Prevent immediate repetition if possible
    if (q.id === lastQuestionId && allQuestions.length > 1) {
      weight = 0;
    }

    return { question: q, weight };
  });

  // Normalize weights
  const totalWeight = candidates.reduce((sum, c) => sum + c.weight, 0);
  
  // If total weight is 0 (e.g. only 1 question and it was just asked), return it anyway
  if (totalWeight === 0) return allQuestions[0];

  let random = Math.random() * totalWeight;
  
  for (const candidate of candidates) {
    if (random < candidate.weight) {
      return candidate.question;
    }
    random -= candidate.weight;
  }

  return candidates[0].question;
}

export function updateStats(
  currentStats: Record<string, QuestionStats>, 
  questionId: string, 
  isCorrect: boolean
): Record<string, QuestionStats> {
  const existing = currentStats[questionId] || {
    questionId,
    streak: 0,
    attempts: 0,
    lastAnswered: 0
  };

  const newStats = { ...currentStats };
  
  newStats[questionId] = {
    ...existing,
    attempts: existing.attempts + 1,
    streak: isCorrect ? existing.streak + 1 : 0, // Reset streak on error
    lastAnswered: Date.now()
  };

  return newStats;
}