'use strict';

jQuery(document).ready(function() {
    const tagsTagTemplate = function (object) {
        return $(
            '<span class="label label-tag" style="background: ' + object.element.dataset.color + ';"> ' + appearanceSanitizeHTML(object.text) + '</span>'
        );
    }

    const tagSelectionTemplate = function (object, container) {
        container[0].style.background = object.element.dataset.color;
        return tagsTagTemplate(object);
    }

    const tagResultTemplate = function (object) {
        if (!object.element) {
            return '';
        }

        return tagsTagTemplate(object);
    }

    $('.select2-tags').select2({
        templateSelection: tagSelectionTemplate,
        templateResult: tagResultTemplate,
        width: '100%'
    });
});
