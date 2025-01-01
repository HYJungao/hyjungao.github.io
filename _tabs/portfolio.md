---
icon: fas fa-briefcase
order: 4
---

<div class="gallery">
    {% for item in site.data.gallery %}
    <div class="gallery-item">
    <img src="{{ item.image }}" alt="{{ item.title }}">
    </div>
    {% endfor %}
</div>

<style>
.gallery {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(5, minmax(150px, auto));
    grid-gap: 2px;
}
.gallery-item {
    position: relative;
    overflow: hidden;
    margin: 0;
    padding: 0;
}
.gallery-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
}
.gallery-item:hover img {
    transform: scale(1.1);
}
</style>