import React, { useState, useEffect } from 'react';
import { Question } from '../types';
import { Check, X, AlertCircle } from 'lucide-react';

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

  return (
    <div className="w-full max-w-2xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden border border-slate-200">
      {/* Header / Category */}
      <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
        <span className="text-xs font-semibold tracking-wider text-slate-500 uppercase">
          {question.category}
        </span>
        <div className="flex items-center gap-2">
           <span className="text-xs text-slate-400">Streak:</span>
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
          <div className="flex items-center w-full justify-between">
            <div className={`flex items-center gap-2 font-medium ${isCorrect ? 'text-green-600' : 'text-red-600'}`}>
                {isCorrect ? (
                    <>
                        <div className="p-1 bg-green-100 rounded-full"><Check size={16} /></div>
                        <span>Correct! Well done.</span>
                    </>
                ) : (
                    <>
                        <div className="p-1 bg-red-100 rounded-full"><X size={16} /></div>
                        <span>Incorrect. Review the answer.</span>
                    </>
                )}
            </div>
            <button
                onClick={handleNext}
                className="px-6 py-2.5 bg-brand-600 hover:bg-brand-700 text-white font-medium rounded-lg shadow-sm transition-colors flex items-center gap-2"
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