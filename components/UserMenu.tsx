import React from 'react';
import { User, signInWithPopup, signOut } from 'firebase/auth';
import { auth, googleProvider } from '../firebase.ts';
import { LogIn, LogOut, User as UserIcon } from 'lucide-react';

interface UserMenuProps {
  user: User | null;
}

const UserMenu: React.FC<UserMenuProps> = ({ user }) => {
  
  const handleLogin = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (error) {
      console.error("Login failed", error);
      alert("Login fallito. Controlla la console per i dettagli.");
    }
  };

  const handleLogout = async () => {
    try {
      if (window.confirm("Sei sicuro di voler uscire?")) {
        await signOut(auth);
        window.location.reload(); // Reload to reset state clean
      }
    } catch (error) {
      console.error("Logout failed", error);
    }
  };

  if (user) {
    return (
      <div className="flex items-center gap-3">
        <div className="hidden sm:flex flex-col items-end">
            <span className="text-sm font-medium text-slate-700 leading-none">{user.displayName}</span>
            <span className="text-[10px] text-slate-400">Sync Active</span>
        </div>
        
        {user.photoURL ? (
            <img src={user.photoURL} alt={user.displayName || "User"} className="w-8 h-8 rounded-full border border-slate-200" />
        ) : (
            <div className="w-8 h-8 rounded-full bg-brand-100 flex items-center justify-center text-brand-600">
                <UserIcon size={16} />
            </div>
        )}

        <button 
            onClick={handleLogout}
            className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-full transition-colors"
            title="Esci"
        >
            <LogOut size={18} />
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={handleLogin}
      className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 rounded-lg text-sm font-medium transition-all shadow-sm"
    >
      <LogIn size={16} />
      <span className="hidden sm:inline">Salva Progressi</span>
    </button>
  );
};

export default UserMenu;