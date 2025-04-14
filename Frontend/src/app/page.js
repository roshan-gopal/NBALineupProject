'use client';

import { useState, useEffect } from 'react';
import { lineupApi } from '../services/api';

const NBA_TEAMS = [
    { id: 'ATL', name: 'Atlanta Hawks' },
    { id: 'BOS', name: 'Boston Celtics' },
    { id: 'BKN', name: 'Brooklyn Nets' },
    { id: 'CHA', name: 'Charlotte Hornets' },
    { id: 'CHI', name: 'Chicago Bulls' },
    { id: 'CLE', name: 'Cleveland Cavaliers' },
    { id: 'DAL', name: 'Dallas Mavericks' },
    { id: 'DEN', name: 'Denver Nuggets' },
    { id: 'DET', name: 'Detroit Pistons' },
    { id: 'GSW', name: 'Golden State Warriors' },
    { id: 'HOU', name: 'Houston Rockets' },
    { id: 'IND', name: 'Indiana Pacers' },
    { id: 'LAC', name: 'Los Angeles Clippers' },
    { id: 'LAL', name: 'Los Angeles Lakers' },
    { id: 'MEM', name: 'Memphis Grizzlies' },
    { id: 'MIA', name: 'Miami Heat' },
    { id: 'MIL', name: 'Milwaukee Bucks' },
    { id: 'MIN', name: 'Minnesota Timberwolves' },
    { id: 'NOP', name: 'New Orleans Pelicans' },
    { id: 'NYK', name: 'New York Knicks' },
    { id: 'OKC', name: 'Oklahoma City Thunder' },
    { id: 'ORL', name: 'Orlando Magic' },
    { id: 'PHI', name: 'Philadelphia 76ers' },
    { id: 'PHX', name: 'Phoenix Suns' },
    { id: 'POR', name: 'Portland Trail Blazers' },
    { id: 'SAC', name: 'Sacramento Kings' },
    { id: 'SAS', name: 'San Antonio Spurs' },
    { id: 'TOR', name: 'Toronto Raptors' },
    { id: 'UTA', name: 'Utah Jazz' },
    { id: 'WAS', name: 'Washington Wizards' }
];

export default function Home() {
    const [homeTeam, setHomeTeam] = useState('NYK');
    const [awayTeam, setAwayTeam] = useState('');
    const [awayTeamRoster, setAwayTeamRoster] = useState([]);
    const [selectedPlayers, setSelectedPlayers] = useState([]);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (awayTeam) {
            fetchAwayTeamRoster();
        }
    }, [awayTeam]);

    const fetchAwayTeamRoster = async () => {
        try {
            const players = await lineupApi.getTeamPlayers(awayTeam);
            setAwayTeamRoster(players);
        } catch (err) {
            setError('Error fetching team roster');
            console.error(err);
        }
    };

    const handlePlayerSelect = (player) => {
        if (selectedPlayers.includes(player)) {
            setSelectedPlayers(selectedPlayers.filter(p => p !== player));
        } else if (selectedPlayers.length < 5) {
            setSelectedPlayers([...selectedPlayers, player]);
        }
    };

    const handleOptimize = async () => {
        if (selectedPlayers.length !== 5) {
            setError('Please select exactly 5 opponent players');
            return;
        }

        setLoading(true);
        setError(null);
        
        try {
            const response = await lineupApi.optimizeLineup(
                selectedPlayers.map(p => p.PLAYER),
                homeTeam
            );
            setResult(response);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-[#fffaf0] to-[#fffdf5]">
            <div className="max-w-7xl mx-auto px-4 py-8">
                <header className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-gray-800 mb-2">NBA Lineup Optimizer</h1>
                    <p className="text-lg text-gray-600">Find your team's best lineup in any situation</p>
                </header>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    <div className="backdrop-blur-xl bg-white/70 p-6 rounded-lg border border-gray-200 shadow-lg">
                        <h2 className="text-xl font-semibold mb-4 text-gray-700">Home Team</h2>
                        <select 
                            value={homeTeam}
                            onChange={(e) => setHomeTeam(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded bg-white/80 text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            {NBA_TEAMS.map(team => (
                                <option key={team.id} value={team.id}>
                                    {team.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="backdrop-blur-xl bg-white/70 p-6 rounded-lg border border-gray-200 shadow-lg">
                        <h2 className="text-xl font-semibold mb-4 text-gray-700">Opponent Team</h2>
                        <select 
                            value={awayTeam}
                            onChange={(e) => {
                                setAwayTeam(e.target.value);
                                setSelectedPlayers([]);
                            }}
                            className="w-full p-2 border border-gray-300 rounded bg-white/80 text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="">Select opponent team</option>
                            {NBA_TEAMS.map(team => (
                                <option key={team.id} value={team.id}>
                                    {team.name}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {awayTeam && (
                    <div className="mb-8 backdrop-blur-xl bg-white/70 p-6 rounded-lg border border-gray-200 shadow-lg">
                        <h2 className="text-xl font-semibold mb-4 text-gray-700">Select Opponent Players (5)</h2>
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                            {awayTeamRoster.map(player => (
                                <div 
                                    key={player.PLAYER}
                                    onClick={() => handlePlayerSelect(player)}
                                    className={`p-4 rounded-lg cursor-pointer transition-all duration-300 border hover:shadow-[0_0_20px_rgba(34,197,94,0.5)] hover:-translate-y-0.5 ${
                                        selectedPlayers.includes(player) 
                                            ? 'bg-blue-50 border-blue-500 text-blue-700' 
                                            : 'bg-white/80 border-gray-200 hover:border-green-400 text-gray-700'
                                    }`}
                                >
                                    <p className="font-medium">{player.PLAYER}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {selectedPlayers.length > 0 && (
                    <div className="mb-8 backdrop-blur-xl bg-white/70 p-6 rounded-lg border border-gray-200 shadow-lg">
                        <h2 className="text-xl font-semibold mb-4 text-gray-700">Selected Lineup ({selectedPlayers.length}/5)</h2>
                        <div className="flex flex-wrap gap-2">
                            {selectedPlayers.map(player => (
                                <div 
                                    key={player.PLAYER}
                                    className="bg-blue-50 text-blue-700 px-4 py-2 rounded-full border border-blue-200"
                                >
                                    {player.PLAYER}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div className="text-center mb-8">
                    <button
                        onClick={handleOptimize}
                        disabled={loading || selectedPlayers.length !== 5}
                        className={`px-8 py-3 rounded-lg text-white font-medium text-lg transition-colors shadow-lg ${
                            loading || selectedPlayers.length !== 5 
                                ? 'bg-gray-400 cursor-not-allowed' 
                                : 'bg-blue-500 hover:bg-blue-600'
                        }`}
                    >
                        {loading ? 'Optimizing...' : 'Find Optimal Lineup'}
                    </button>
                </div>

                {error && (
                    <div className="mb-8 p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 backdrop-blur-xl shadow-lg">
                        {error}
                    </div>
                )}

                {result && (
                    <div className="backdrop-blur-xl bg-white/70 p-6 rounded-lg border border-gray-200 shadow-lg">
                        <h2 className="text-2xl font-bold mb-6 text-gray-800">
                            Top Lineups for {NBA_TEAMS.find(team => team.id === homeTeam)?.name}
                        </h2>
                        <div className="overflow-x-auto mb-8">
                            <table className="w-full text-gray-700">
                                <thead className="text-left border-b-2 border-gray-300">
                                    <tr>
                                        <th className="py-4 px-6 text-lg font-semibold">Rank</th>
                                        <th className="py-4 px-6 text-lg font-semibold">Lineup</th>
                                        <th className="py-4 px-6 text-lg font-semibold">Model Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {result.lineups.slice(0, 3).map((lineup, index) => (
                                        <tr key={index} className={`transition-all duration-300 hover:shadow-[0_0_25px_rgba(34,197,94,0.6)] hover:-translate-y-0.5 ${index === 0 ? "bg-blue-50/50" : "hover:bg-gray-50"}`}>
                                            <td className="py-4 px-6 font-medium">{index + 1}</td>
                                            <td className="py-4 px-6 font-medium">{lineup.lineup.join(", ")}</td>
                                            <td className="py-4 px-6 font-medium">{lineup.score.toFixed(2)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        <div className="mt-8 p-6 bg-gray-50/50 rounded-lg border border-gray-200">
                            <h3 className="text-xl font-bold text-gray-800 mb-4">Opponent Lineup</h3>
                            <p className="text-lg text-gray-700 font-medium">{selectedPlayers.map(p => p.PLAYER).join(", ")}</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}