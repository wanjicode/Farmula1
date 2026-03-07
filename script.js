// Sticky Navbar
window.addEventListener('scroll', function() {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('sticky');
    } else {
        navbar.classList.remove('sticky');
    }
});

// Intersection Observer for scroll animations
const faders = document.querySelectorAll('.fade-in');

const appearOptions = {
    threshold: 0.15,
    rootMargin: "0px 0px -50px 0px"
};

const appearOnScroll = new IntersectionObserver(function(entries, observer) {
    entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        
        entry.target.classList.add('appear');
        observer.unobserve(entry.target);
    });
}, appearOptions);

faders.forEach(fader => {
    appearOnScroll.observe(fader);
});

// Language Toggle (Basic Implementation)
let currentLang = 'en';
const langBtn = document.getElementById('lang-toggle');

langBtn.addEventListener('click', function() {
    currentLang = currentLang === 'en' ? 'sw' : 'en';
    
    // Update elements with data attributes
    const elements = document.querySelectorAll('[data-en][data-sw]');
    
    elements.forEach(el => {
        el.innerText = el.getAttribute(`data-${currentLang}`);
    });
});