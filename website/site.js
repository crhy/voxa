// Intentionally tiny: the site works fully without JavaScript.
const nav = document.querySelector('.nav');
addEventListener('scroll', () => nav.classList.toggle('scrolled', scrollY > 20), {passive:true});
