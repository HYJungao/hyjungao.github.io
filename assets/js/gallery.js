document.addEventListener("DOMContentLoaded", function () {
    const items = document.querySelectorAll('.gallery-item');
    items.forEach(item => {
      const randomColumn = Math.floor(Math.random() * 2) + 1;
      const randomRow = Math.floor(Math.random() * 2) + 1;
      item.style.gridColumn = `span ${randomColumn}`;
      item.style.gridRow = `span ${randomRow}`;
    });
  });
  