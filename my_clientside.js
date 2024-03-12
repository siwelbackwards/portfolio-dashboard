if(!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.clientside = {
    update_timer: function(n_intervals) {
        const calculateTimeLeft = () => {
            const targetDate = new Date("2024-09-01").getTime(); // Set your target date here
            const now = new Date().getTime();
            const difference = targetDate - now;

            let timeLeft = {};

            if (difference > 0) {
                timeLeft = {
                    days: Math.floor(difference / (1000 * 60 * 60 * 24)),
                    hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
                    minutes: Math.floor((difference / 1000 / 60) % 60),
                    seconds: Math.floor((difference / 1000) % 60)
                };
            }

            return timeLeft;
        };

        const timeLeft = calculateTimeLeft();
        if(Object.keys(timeLeft).length > 0) {
            return `Time until portfolio reshuffle: ${timeLeft.days} days, ${timeLeft.hours}:${timeLeft.minutes}:${timeLeft.seconds}`;
        } else {
            return "Time for portfolio reshuffle!";
        }
    }
}
