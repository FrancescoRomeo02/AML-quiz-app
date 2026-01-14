import React, { useMemo } from 'react';
import { QuestionStats } from '../types';
import { Trophy, Activity, RotateCcw } from 'lucide-react';
import { User } from 'firebase/auth';

interface DashboardProps {
  totalQuestions: number;
  stats: Record<string, QuestionStats>;
  onStart: () => void;
  onReset: () => void;
  user: User | null;
}

const Dashboard: React.FC<DashboardProps> = ({ totalQuestions, stats, onStart, onReset, user }) => {
  
  const dashboardStats = useMemo(() => {
    const attempted = Object.keys(stats).length;
    const statValues = Object.values(stats);
    const mastered = statValues.filter(s => s.streak >= 3).length;
    const learning = statValues.filter(s => s.streak > 0 && s.streak < 3).length;
    const struggling = statValues.filter(s => s.streak === 0).length;
    
    return { total: totalQuestions, attempted, mastered, learning, struggling };
  }, [totalQuestions, stats]);

  return (
    <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
      <div className="text-center space-y-3">
          <h2 className="text-3xl font-bold text-slate-900">Spaced Repetition Training</h2>
          <p className="text-slate-500 text-lg">
              The system adapts to your performance, showing you difficult questions more often until you master them.
          </p>
          {!user && (
              <div className="p-3 bg-blue-50 text-blue-800 rounded-lg text-sm inline-block">
                  💡 Tip: Effettua il login per salvare i progressi nel cloud e non perderli se cancelli la cache.
              </div>
          )}
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
              onClick={onStart}
              className="flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white text-lg font-semibold px-8 py-4 rounded-full shadow-lg shadow-brand-500/30 transition-all hover:scale-105"
          >
              <Activity className="w-5 h-5" />
              {dashboardStats.attempted > 0 ? 'Continue Training' : 'Start Training'}
          </button>
      </div>
      
      {dashboardStats.attempted > 0 && (
          <div className="text-center">
              <button 
                  onClick={onReset}
                  className="text-sm text-slate-400 hover:text-red-500 flex items-center gap-1 mx-auto transition-colors"
              >
                  <RotateCcw size={14} /> Reset Progress
              </button>
          </div>
      )}
    </div>
  );
};

export default Dashboard;