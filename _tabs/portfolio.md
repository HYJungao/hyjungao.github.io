---
icon: fas fa-briefcase
order: 4
---

<script src="../assets/js/gallery.js"></script>

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
    grid-template-columns: repeat(auto, minmax(150px, 1fr));
    grid - gap: 2px;
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
    object - fit: cover;
    t
    ransition: transform 0.3s ease;
}
.gallery-item:hover img {
    transform: scale(1.1);
}
</style>