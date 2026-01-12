import { useState, useEffect, useMemo } from 'react';
import { questions as questionsData } from './questions';
import { Question, QuestionStats } from './types';
import { getNextQuestion, updateStats } from './utils/srs';
import QuizCard from './components/QuizCard';
import { Brain, Trophy, Activity, RotateCcw } from 'lucide-react';

export default function App() {
  const [questions, setQuestions] = useState<Question[]>([]);
  
  // Initialize stats from localStorage if available
  const [stats, setStats] = useState<Record<string, QuestionStats>>(() => {
    try {
      const saved = localStorage.getItem('ml-mastery-stats');
      return saved ? JSON.parse(saved) : {};
    } catch (e) {
      console.error("Failed to load stats from storage", e);
      return {};
    }
  });

  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [loading, setLoading] = useState(true);

  // Load questions on mount
  useEffect(() => {
    // Cast JSON data to Question type to ensure compatibility
    setQuestions(questionsData as Question[]);
    setLoading(false);
  }, []);

  // Save stats to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('ml-mastery-stats', JSON.stringify(stats));
  }, [stats]);

  const startSession = () => {
    setSessionActive(true);
    const next = getNextQuestion(questions, stats, null);
    setCurrentQuestion(next);
  };

  const handleAnswer = (isCorrect: boolean) => {
    if (!currentQuestion) return;

    // Update stats
    const newStats = updateStats(stats, currentQuestion.id, isCorrect);
    setStats(newStats);

    // Get next question (delay slightly for UX if needed, but here we switch immediately on click)
    const next = getNextQuestion(questions, newStats, currentQuestion.id);
    setCurrentQuestion(next);
  };

  const handleReset = () => {
    if (window.confirm("Sei sicuro di voler resettare tutti i progressi? Questa azione è irreversibile.")) {
      setStats({});
    }
  };

  // Computed Stats for Dashboard
  const dashboardStats = useMemo(() => {
    const total = questions.length;
    const attempted = Object.keys(stats).length;
    // Explicitly cast to QuestionStats[] to handle cases where Object.values returns unknown[]
    const statValues = Object.values(stats) as QuestionStats[];
    const mastered = statValues.filter(s => s.streak >= 3).length;
    const learning = statValues.filter(s => s.streak > 0 && s.streak < 3).length;
    const struggling = statValues.filter(s => s.streak === 0).length;
    
    return { total, attempted, mastered, learning, struggling };
  }, [questions, stats]);

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center text-slate-500">Loading curriculum...</div>;
  }

  return (
    <div className="min-h-screen bg-slate-50 pb-20">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center gap-2 text-brand-600">
                <Brain className="w-6 h-6" />
                <h1 className="font-bold text-lg tracking-tight text-slate-900">ML Mastery</h1>
            </div>
            {sessionActive && (
                 <button 
                 onClick={() => setSessionActive(false)}
                 className="text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors"
               >
                 Exit Session
               </button>
            )}
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        {!sessionActive ? (
          // DASHBOARD
          <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
            <div className="text-center space-y-3">
                <h2 className="text-3xl font-bold text-slate-900">Spaced Repetition Training</h2>
                <p className="text-slate-500 text-lg">
                    The system adapts to your performance, showing you difficult questions more often until you master them.
                </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 text-center">
                    <div className="text-2xl font-bold text-slate-900">{dashboardStats.total}</div>
                    <div className="text-xs text-slate-500 font-medium uppercase mt-1">Total Questions</div>
                </div>
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 text-center">
                    <div className="text-2xl font-bold text-green-600">{dashboardStats.mastered}</div>
                    <div className="text-xs text-slate-500 font-medium uppercase mt-1">Mastered</div>
                </div>
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 text-center">
                    <div className="text-2xl font-bold text-amber-500">{dashboardStats.learning}</div>
                    <div className="text-xs text-slate-500 font-medium uppercase mt-1">Learning</div>
                </div>
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 text-center">
                    <div className="text-2xl font-bold text-red-500">{dashboardStats.struggling}</div>
                    <div className="text-xs text-slate-500 font-medium uppercase mt-1">Needs Review</div>
                </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <Trophy className="w-5 h-5 text-brand-500" />
                    Your Progress
                </h3>
                <div className="w-full bg-slate-100 rounded-full h-4 overflow-hidden flex">
                    <div 
                        className="bg-green-500 h-full transition-all duration-500" 
                        style={{ width: `${(dashboardStats.mastered / dashboardStats.total) * 100}%` }} 
                        title="Mastered"
                    />
                    <div 
                        className="bg-amber-400 h-full transition-all duration-500" 
                        style={{ width: `${(dashboardStats.learning / dashboardStats.total) * 100}%` }} 
                        title="Learning"
                    />
                    <div 
                         className="bg-slate-200 h-full"
                         style={{ flex: 1 }}
                    />
                </div>
                <div className="flex justify-between text-sm text-slate-500 mt-2">
                    <span>0%</span>
                    <span>100% Mastery</span>
                </div>
            </div>

            <div className="flex justify-center pt-4">
                <button 
                    onClick={startSession}
                    className="flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white text-lg font-semibold px-8 py-4 rounded-full shadow-lg shadow-brand-500/30 transition-all hover:scale-105"
                >
                    <Activity className="w-5 h-5" />
                    {dashboardStats.attempted > 0 ? 'Continue Training' : 'Start Training'}
                </button>
            </div>
            
            {dashboardStats.attempted > 0 && (
                <div className="text-center">
                    <button 
                        onClick={handleReset}
                        className="text-sm text-slate-400 hover:text-red-500 flex items-center gap-1 mx-auto transition-colors"
                    >
                        <RotateCcw size={14} /> Reset Progress
                    </button>
                </div>
            )}
          </div>
        ) : (
            // QUIZ MODE
          <div className="animate-fade-in-up">
            {currentQuestion && (
                <QuizCard 
                    question={currentQuestion} 
                    onAnswer={handleAnswer}
                    streak={stats[currentQuestion.id]?.streak || 0}
                />
            )}
          </div>
        )}
      </main>
    </div>
  );
}