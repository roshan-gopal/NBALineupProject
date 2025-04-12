'use client'; // This is needed because we're using hooks

import { useState } from 'react';
import { lineupApi } from '../services/api';

export default function Home() {
    const [opponentPlayers, setOpponentPlayers] = useState('');
    const [team, setTeam] = useState('NYK');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleOptimize = async () => {
        setLoading(true);
        setError(null);
        try {
            // Split the input string into an array of players and clean up
            const players = opponentPlayers.split(',')
                .map(player => player.trim())
                .filter(player => player.length > 0);
            
            console.log('Raw input:', opponentPlayers);
            console.log('Processed players:', players);
            console.log('Team:', team);
            
            if (players.length !== 5) {
                throw new Error('Please enter exactly 5 opponent players, separated by commas');
            }
            
            // Call the API with the processed data
            const response = await lineupApi.optimizeLineup(players, team);
            setResult(response);
        } catch (err) {
            console.error('Error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto p-4">
            <h1 className="text-2xl font-bold mb-4">NBA Lineup Optimizer</h1>
            
            <div className="mb-4">
                <label className="block mb-2">
                    Opponent Players (comma-separated):
                    <input
                        type="text"
                        value={opponentPlayers}
                        onChange={(e) => setOpponentPlayers(e.target.value)}
                        className="w-full p-2 border rounded"
                        placeholder="Jayson Tatum, Jaylen Brown, Kristaps Porzingis, Derrick White, Jrue Holiday"
                    />
                </label>
            </div>

            <div className="mb-4">
                <label className="block mb-2">
                    Team:
                    <input
                        type="text"
                        value={team}
                        onChange={(e) => setTeam(e.target.value)}
                        className="w-full p-2 border rounded"
                        placeholder="NYK"
                    />
                </label>
            </div>

            <button
                onClick={handleOptimize}
                disabled={loading}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
            >
                {loading ? 'Optimizing...' : 'Optimize Lineup'}
            </button>

            {error && (
                <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
                    Error: {error.toString()}
                </div>
            )}

            {result && (
                <div className="mt-4 p-4 bg-gray-100 rounded text-black">
                    <h2 className="text-xl font-semibold mb-2">Optimized Lineup</h2>
                    <div className="mb-2">
                        <strong>Lineup:</strong> {Array.isArray(result.lineup) ? result.lineup.join(', ') : 'Invalid lineup data'}
                    </div>
                    <div>
                        <strong>Opponent:</strong> {Array.isArray(result.opponent) ? result.opponent.join(', ') : 'Invalid opponent data'}
                    </div>
                </div>
            )}
        </div>
    );
}
