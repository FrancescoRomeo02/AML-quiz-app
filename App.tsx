import { useState } from 'react';
import { questions as questionsData } from './questions'; // Using the TS source of truth
import { Question } from './types';
import { getNextQuestion } from './utils/srs';
import { useProgress } from './hooks/useProgress';

// Components
import QuizCard from './components/QuizCard';
import UserMenu from './components/UserMenu';
import Dashboard from './components/Dashboard';
import { Brain } from 'lucide-react';

export default function App() {
  const { stats, user, loading, updateProgress, resetProgress } = useProgress();
  
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [sessionActive, setSessionActive] = useState(false);

  const startSession = () => {
    setSessionActive(true);
    const next = getNextQuestion(questionsData as Question[], stats, null);
    setCurrentQuestion(next);
  };

  const handleAnswer = (isCorrect: boolean) => {
    if (!currentQuestion) return;

    // 1. Update stats (Local + Cloud via Hook)
    const newStats = updateProgress(currentQuestion.id, isCorrect);

    // 2. Get next question based on NEW stats
    const next = getNextQuestion(questionsData as Question[], newStats, currentQuestion.id);
    setCurrentQuestion(next);
  };

  if (loading) {
    return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="animate-pulse text-slate-400 font-medium">Loading AML Mastery...</div>
        </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 pb-20">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center gap-2 text-brand-600 select-none cursor-pointer" onClick={() => setSessionActive(false)}>
                <Brain className="w-6 h-6" />
                <h1 className="font-bold text-lg tracking-tight text-slate-900 hidden sm:block">AML Mastery</h1>
                <h1 className="font-bold text-lg tracking-tight text-slate-900 sm:hidden">AML</h1>
            </div>
            
            <div className="flex items-center gap-4">
                {sessionActive && (
                     <button 
                     onClick={() => setSessionActive(false)}
                     className="text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors"
                   >
                     Exit
                   </button>
                )}
                
                <UserMenu user={user} />
            </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        {!sessionActive ? (
          <Dashboard 
            totalQuestions={questionsData.length}
            stats={stats}
            onStart={startSession}
            onReset={resetProgress}
            user={user}
          />
        ) : (
          <div className="animate-fade-in-up">
            {currentQuestion ? (
                <QuizCard 
                    question={currentQuestion} 
                    onAnswer={handleAnswer}
                    streak={stats[currentQuestion.id]?.streak || 0}
                />
            ) : (
                <div className="text-center text-slate-500 mt-10">
                    No questions available.
                </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}