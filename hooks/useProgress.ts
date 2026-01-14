import { useState, useEffect, useRef } from 'react';
import { QuestionStats } from '../types';
import { updateStats } from '../utils/srs';
import { auth, db } from '../firebase';
import { onAuthStateChanged, User } from 'firebase/auth';
import { doc, getDoc, setDoc } from 'firebase/firestore';

export function useProgress() {
  // Initialize stats from localStorage
  const [stats, setStats] = useState<Record<string, QuestionStats>>(() => {
    try {
      const saved = localStorage.getItem('ml-mastery-stats');
      return saved ? JSON.parse(saved) : {};
    } catch (e) {
      console.error("Failed to load stats from storage", e);
      return {};
    }
  });

  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isCloudSynced, setIsCloudSynced] = useState(false);

  // Ref to access current stats inside async callbacks without dependency loops
  const statsRef = useRef(stats);
  useEffect(() => { statsRef.current = stats; }, [stats]);

  // Auth & Cloud Sync Logic
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setUser(currentUser);
      
      if (currentUser) {
        // Sync logic
        const userDocRef = doc(db, 'users', currentUser.uid);
        try {
            const docSnap = await getDoc(userDocRef);
            if (docSnap.exists()) {
                // Restore from Cloud
                console.log("Cloud data found, syncing...");
                setStats(docSnap.data().stats || {});
            } else {
                // Upload Local to Cloud (First time)
                console.log("Migrating local data to cloud...");
                await setDoc(userDocRef, { 
                    stats: statsRef.current,
                    lastUpdated: Date.now()
                });
            }
            setIsCloudSynced(true);
        } catch (error) {
            console.error("Error syncing with cloud:", error);
        }
      } else {
          setIsCloudSynced(false);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  // Persistence Logic (Local + Cloud)
  useEffect(() => {
    localStorage.setItem('ml-mastery-stats', JSON.stringify(stats));

    if (user && isCloudSynced) {
        const saveToCloud = async () => {
            try {
                const userDocRef = doc(db, 'users', user.uid);
                await setDoc(userDocRef, { 
                    stats, 
                    lastUpdated: Date.now() 
                }, { merge: true });
            } catch (e) {
                console.error("Failed to save to cloud", e);
            }
        };
        saveToCloud();
    }
  }, [stats, user, isCloudSynced]);

  // Public Methods
  const handleAnswerUpdate = (questionId: string, isCorrect: boolean) => {
    const newStats = updateStats(stats, questionId, isCorrect);
    setStats(newStats);
    return newStats;
  };

  const resetProgress = () => {
    if (window.confirm("Sei sicuro di voler resettare tutti i progressi? Questa azione è irreversibile.")) {
      setStats({});
    }
  };

  return {
    stats,
    user,
    loading,
    updateProgress: handleAnswerUpdate,
    resetProgress
  };
}