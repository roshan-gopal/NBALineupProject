const API_BASE_URL = 'http://localhost:8000/api';

export const lineupApi = {
    optimizeLineup: async (opponentPlayers, team) => {
        try {
            // Construct URL with query parameter
            const url = `${API_BASE_URL}/optimize?team=${encodeURIComponent(team)}`;
            
            // Detailed URL logging
            console.log('URL Details:');
            console.log('Base URL:', API_BASE_URL);
            console.log('Team parameter:', team);
            console.log('Encoded team parameter:', encodeURIComponent(team));
            console.log('Full URL:', url);
            
            console.log('API Request Details:');
            console.log('Method: POST');
            console.log('Headers:', {
                'Content-Type': 'application/json'
            });
            console.log('Request Body (opponent players):', JSON.stringify(opponentPlayers, null, 2));
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Send opponent players directly as an array
                body: JSON.stringify(opponentPlayers)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Backend error details:', {
                    status: response.status,
                    statusText: response.statusText,
                    errorData: errorData
                });
                throw new Error(JSON.stringify(errorData.detail) || `HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Backend response:', JSON.stringify(result, null, 2));
            return result;
        } catch (error) {
            console.error('Error optimizing lineup:', error);
            throw error;
        }
    },

    getTeamPlayers: async (team) => {
        try {
            const response = await fetch(`${API_BASE_URL}/team/${team}`);
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching team players:', error);
            throw error;
        }
    },

    testOptimization: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/test`);
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error testing optimization:', error);
            throw error;
        }
    }
}; 