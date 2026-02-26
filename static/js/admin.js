const selectAllCheckbox = document.getElementById('selectAll');
const rowCheckboxes = document.querySelectorAll('.row-checkbox');
const deleteBtn = document.getElementById('deleteBtn');
const selectedCount = document.getElementById('selectedCount');
const selectedText = document.getElementById('selectedText');
const deleteForm = document.getElementById('deleteForm');

function updateSelectionUI() {
  const checked = document.querySelectorAll('.row-checkbox:checked');
  const count = checked.length;
  selectedCount.textContent = count;
  deleteBtn.disabled = count === 0;

  if (count > 0) {
    selectedText.textContent = count + ' item' + (count > 1 ? 's' : '') + ' selected';
  } else {
    selectedText.textContent = '';
  }

  if (rowCheckboxes.length > 0) {
    selectAllCheckbox.checked = count === rowCheckboxes.length;
    selectAllCheckbox.indeterminate = count > 0 && count < rowCheckboxes.length;
  }
}

if (selectAllCheckbox) {
  selectAllCheckbox.addEventListener('change', function() {
    rowCheckboxes.forEach(cb => { cb.checked = this.checked; });
    updateSelectionUI();
  });
}

rowCheckboxes.forEach(checkbox => {
  checkbox.addEventListener('change', updateSelectionUI);
});

if (deleteForm) {
  deleteForm.addEventListener('submit', function(e) {
    const checked = document.querySelectorAll('.row-checkbox:checked');
    if (checked.length === 0) {
      e.preventDefault();
      return false;
    }
    if (!confirm('Are you sure you want to delete ' + checked.length + ' selected item(s)? This cannot be undone.')) {
      e.preventDefault();
      return false;
    }
  });
}

updateSelectionUI();
