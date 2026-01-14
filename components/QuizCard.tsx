import React, { useState, useEffect } from 'react';
import { Question } from '../types';
import { Check, X, AlertCircle, Flag, AlertTriangle } from 'lucide-react';
import { REPORT_EMAIL, GITHUB_REPO } from '../config';

interface QuizCardProps {
  question: Question;
  onAnswer: (isCorrect: boolean) => void;
  streak: number;
}

const QuizCard: React.FC<QuizCardProps> = ({ question, onAnswer, streak }) => {
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [isCorrect, setIsCorrect] = useState(false);

  // Reset state when question changes
  useEffect(() => {
    setSelectedIndices([]);
    setHasSubmitted(false);
    setIsCorrect(false);
  }, [question.id]);

  const isMultiSelect = question.correctIndices.length > 1;

  const toggleOption = (index: number) => {
    if (hasSubmitted) return;

    if (isMultiSelect) {
      setSelectedIndices(prev => 
        prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
      );
    } else {
      setSelectedIndices([index]);
    }
  };

  const handleSubmit = () => {
    if (selectedIndices.length === 0) return;

    // Check correctness
    // Sort both arrays to compare
    const sortedSelected = [...selectedIndices].sort();
    const sortedCorrect = [...question.correctIndices].sort();
    
    const correct = JSON.stringify(sortedSelected) === JSON.stringify(sortedCorrect);
    
    setIsCorrect(correct);
    setHasSubmitted(true);
  };

  const handleNext = () => {
    onAnswer(isCorrect);
  };

  const handleReport = () => {
    const title = `Segnalazione Errore: Domanda ${question.id}`;
    const body = `
**ID Domanda:** ${question.id}
**Categoria:** ${question.category}
**Testo Domanda:** ${question.text.substring(0, 150)}${question.text.length > 150 ? '...' : ''}

**Descrizione dell'errore:**
[Scrivi qui cosa c'è di sbagliato...]
    `.trim();

    if (GITHUB_REPO) {
        // Open GitHub Issue
        const url = `https://github.com/${GITHUB_REPO}/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
        window.open(url, '_blank');
    } else {
        // Fallback to Email
        const url = `mailto:${REPORT_EMAIL}?subject=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
        window.open(url, '_blank');
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden border border-slate-200">
      {/* Header / Category */}
      <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
        <span className="text-xs font-semibold tracking-wider text-slate-500 uppercase truncate max-w-[50%]">
          {question.category}
        </span>
        <div className="flex items-center gap-3">
           {/* Report Button (Icon only) */}
           <button 
             onClick={handleReport}
             className="text-slate-400 hover:text-red-500 transition-colors p-1"
             title="Segnala un errore in questa domanda"
           >
             <Flag size={16} />
           </button>
           
           <div className="h-4 w-px bg-slate-300 mx-1"></div>

           <span className="text-xs text-slate-400 hidden sm:inline">Streak:</span>
           <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${streak > 2 ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
             {streak} 🔥
           </span>
        </div>
      </div>

      <div className="p-6">
        {/* Question Text */}
        <h2 className="text-lg md:text-xl font-medium text-slate-800 mb-6 leading-relaxed">
          {question.text}
        </h2>

        {/* Note for multi-select */}
        {isMultiSelect && (
          <p className="text-xs text-brand-600 font-medium mb-4 flex items-center gap-1">
            <AlertCircle size={14} /> Select all valid options
          </p>
        )}

        {/* Options */}
        <div className="space-y-3">
          {question.options.map((option, idx) => {
            const isSelected = selectedIndices.includes(idx);
            const isAnswerCorrect = question.correctIndices.includes(idx);
            
            // Determine styles based on state
            let containerClass = "relative flex items-start p-4 border rounded-lg cursor-pointer transition-all duration-200 ";
            
            if (hasSubmitted) {
                if (isAnswerCorrect) {
                    containerClass += "bg-green-50 border-green-500 ring-1 ring-green-500 ";
                } else if (isSelected && !isAnswerCorrect) {
                    containerClass += "bg-red-50 border-red-500 ";
                } else {
                    containerClass += "opacity-60 border-slate-200 ";
                }
            } else {
                if (isSelected) {
                    containerClass += "bg-brand-50 border-brand-500 ring-1 ring-brand-500 shadow-sm ";
                } else {
                    containerClass += "hover:bg-slate-50 border-slate-200 ";
                }
            }

            return (
              <div 
                key={idx} 
                onClick={() => toggleOption(idx)}
                className={containerClass}
              >
                <div className="flex-shrink-0 mt-0.5 mr-3">
                  <div className={`w-5 h-5 rounded flex items-center justify-center border ${
                    hasSubmitted && isAnswerCorrect ? 'bg-green-500 border-green-500' : 
                    isSelected ? 'bg-brand-600 border-brand-600' : 'border-slate-300 bg-white'
                  }`}>
                    {(isSelected || (hasSubmitted && isAnswerCorrect)) && (
                      <Check size={14} className="text-white" />
                    )}
                  </div>
                </div>
                <div className="text-sm md:text-base text-slate-700">
                    {option}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Actions */}
      <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex justify-end">
        {!hasSubmitted ? (
          <button
            onClick={handleSubmit}
            disabled={selectedIndices.length === 0}
            className="px-6 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg shadow-sm transition-colors"
          >
            Submit Answer
          </button>
        ) : (
          <div className="flex flex-col sm:flex-row items-center w-full justify-between gap-4">
             {/* Feedback Area */}
             <div className="flex flex-col w-full sm:w-auto">
                <div className={`flex items-center gap-2 font-medium ${isCorrect ? 'text-green-600' : 'text-red-600'}`}>
                    {isCorrect ? (
                        <>
                            <div className="p-1 bg-green-100 rounded-full"><Check size={16} /></div>
                            <span className="text-sm md:text-base">Correct! Well done.</span>
                        </>
                    ) : (
                        <>
                            <div className="p-1 bg-red-100 rounded-full"><X size={16} /></div>
                            <span className="text-sm md:text-base">Incorrect. Review answer.</span>
                        </>
                    )}
                </div>

                {/* Explicit Report Action for Incorrect Answers */}
                {!isCorrect && (
                   <button 
                     onClick={handleReport}
                     className="mt-2 text-xs text-red-500 hover:text-red-700 hover:bg-red-50 px-2 py-1 -ml-2 rounded transition-colors flex items-center gap-1.5 w-fit"
                   >
                     <AlertTriangle size={12} />
                     <span>Pensi sia corretta? Segnala errore</span>
                   </button>
                )}
             </div>

            <button
                onClick={handleNext}
                className="w-full sm:w-auto px-6 py-2.5 bg-brand-600 hover:bg-brand-700 text-white font-medium rounded-lg shadow-sm transition-colors flex items-center justify-center gap-2"
            >
                Next Question
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuizCard;