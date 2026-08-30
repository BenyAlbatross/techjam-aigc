// models-extra.js extends the catalog after models.js creates the controls.
// Add any families introduced by the extension before the final render.
[...new Set(models.map(model => model.family))].sort().forEach(family => {
  const alreadyPresent = [...familyFilter.options].some(option => option.value === family);
  if (!alreadyPresent) {
    const option = document.createElement("option");
    option.value = family;
    option.textContent = family;
    familyFilter.append(option);
  }
});

render();
